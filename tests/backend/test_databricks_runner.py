"""DatabricksRunner's translation layer, exercised against a fake workspace.

No network is involved: the SDK client is injected, so these assert what
the Runner actually sends and how it interprets what comes back — which is
the part that has to be right before any real workspace is available.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.orchestration.databricks_runner import DatabricksRunner, map_run_state
from app.schemas.compute import ComputeSelection
from app.orchestration.schemas import JobStatus, PipelineExecutionRequest


class _FakeFiles:
    """A flat `{path: bytes}` map standing in for a UC Volume.

    `list_directory_contents`/`delete`/`delete_directory` derive directory
    structure from path prefixes rather than modelling a real filesystem —
    enough fidelity for `DatabricksRunner`'s recursive cleanup, which only
    ever lists one level, deletes files, then recurses into subdirectories.
    """

    def __init__(self) -> None:
        self.uploaded: dict[str, bytes] = {}

    def upload(self, file_path, contents, overwrite=None):
        self.uploaded[file_path] = contents.read()

    def download(self, file_path):
        if file_path not in self.uploaded:
            raise _not_found(file_path)
        return SimpleNamespace(contents=SimpleNamespace(read=lambda: self.uploaded[file_path]))

    def list_directory_contents(self, path):
        prefix = path.rstrip("/") + "/"
        children: dict[str, bool] = {}  # child path -> is_directory
        for key in self.uploaded:
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix) :]
            if "/" in remainder:
                child = prefix + remainder.split("/", 1)[0]
                children[child] = True
            else:
                children[key] = False
        if not children:
            raise _not_found(path)
        return [SimpleNamespace(path=child, is_directory=is_dir) for child, is_dir in children.items()]

    def delete(self, file_path):
        if file_path not in self.uploaded:
            raise _not_found(file_path)
        del self.uploaded[file_path]

    def delete_directory(self, dir_path):
        prefix = dir_path.rstrip("/") + "/"
        if any(key.startswith(prefix) for key in self.uploaded):
            raise RuntimeError(f"directory not empty: {dir_path}")
        # Deleting an already-empty (or never-existent) directory succeeds
        # silently, matching the real API — this is what makes repeated
        # cleanup idempotent.


def _not_found(path):
    try:
        from databricks.sdk.errors import NotFound

        return NotFound(path)
    except ImportError:  # pragma: no cover - dependency is declared
        return FileNotFoundError(path)



# The engine arguments the first task actually receives, as a {flag: value}
# map. The task definition carries `{{job.parameters.x}}` placeholders and
# run_now supplies the values, so this resolves them the way Databricks does.
def _submitted_parameters(workspace, task_index=0):
    task = workspace.jobs.submitted_tasks[task_index]
    supplied = workspace.jobs.submitted_parameters or {}
    args = [
        supplied.get(arg[len("{{job.parameters.") : -len("}}")], arg)
        if arg.startswith("{{job.parameters.")
        else arg
        for arg in task.python_wheel_task.parameters
    ]
    flags = {}
    index = 0
    while index < len(args):
        if args[index].startswith("--"):
            has_value = index + 1 < len(args) and not args[index + 1].startswith("--")
            flags[args[index].lstrip("-").replace("-", "_")] = args[index + 1] if has_value else True
            index += 2 if has_value else 1
        else:
            index += 1
    return flags


class _FakeJobs:
    """The runner owns one named Job: it creates it on first use, resets it
    before each run (compute is per-run), then run_now()s it."""

    def __init__(self, state=None) -> None:
        self.create_calls: list[dict] = []
        self.reset_calls: list[dict] = []
        self.run_now_calls: list[dict] = []
        self.cancelled: list[int] = []
        self._jobs: dict[int, str] = {}
        self._state = state or SimpleNamespace(life_cycle_state="RUNNING", result_state=None, state_message="")

    def create(self, access_control_list=None, **settings):
        job_id = 4242 + len(self.create_calls)
        self.create_calls.append({**settings, "access_control_list": access_control_list})
        self._jobs[job_id] = settings.get("name")
        return SimpleNamespace(job_id=job_id)

    def list(self, name=None, **kwargs):
        return [
            SimpleNamespace(job_id=job_id, settings=SimpleNamespace(name=job_name))
            for job_id, job_name in self._jobs.items()
            if name is None or job_name == name
        ]

    def reset(self, job_id=None, new_settings=None):
        self.reset_calls.append({"job_id": job_id, "new_settings": new_settings})

    def run_now(self, job_id=None, job_parameters=None, **kwargs):
        self.run_now_calls.append({"job_id": job_id, "job_parameters": job_parameters})
        return SimpleNamespace(run_id=99)

    def get_run(self, run_id, **kwargs):
        return SimpleNamespace(state=self._state, run_duration=12_000)

    def cancel_run(self, run_id):
        self.cancelled.append(run_id)

    # --- what the tests assert against -------------------------------
    #
    # The job's definition as it stood for the most recent run, whether
    # that came from create() or a later reset().
    @property
    def submitted_settings(self) -> dict:
        if self.reset_calls:
            new = self.reset_calls[-1]["new_settings"]
            return {f.name: getattr(new, f.name) for f in dataclasses.fields(new)}
        return self.create_calls[-1]

    @property
    def submitted_tasks(self) -> list:
        return self.submitted_settings["tasks"]

    @property
    def submitted_parameters(self) -> dict:
        return self.run_now_calls[-1]["job_parameters"]


class _FakeWorkspace:
    def __init__(self, state=None) -> None:
        self.files = _FakeFiles()
        self.jobs = _FakeJobs(state)


@pytest.fixture
def settings(tmp_path, mlflow_db):
    return Settings(
        execution_mode="databricks",
        databricks_host="https://example.invalid",
        databricks_token="test-token",
        databricks_uploads_volumes_root="/Volumes/forecastiq/forecasting/upload_files",
        upload_dir=str(tmp_path / "uploads"),
        mlflow_tracking_uri=f"sqlite:///{mlflow_db}",
    )


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("date,store,sales\n2024-01-01,1,10\n")
    return path


def _request(dataset):
    return PipelineExecutionRequest(
        compute=ComputeSelection(mode="existing_compute", cluster_id="test-cluster"),
        run_id="dbx-run-test",
        dataset_path=str(dataset),
        dataset_name="sales.csv",
        forecast_configuration={
            "date_column": "date",
            "target_column": "sales",
            "key_columns": ["store"],
            "feature_columns": [],
            "aggregation_method": "sum",
        },
        selected_models=["prophet", "xgboost"],
        fallback_model="xgboost",
        horizon=18,
    )


# ---------------------------------------------------------------------
# State translation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "life_cycle,result,expected",
    [
        ("QUEUED", None, JobStatus.PENDING),
        ("PENDING", None, JobStatus.PENDING),
        ("RUNNING", None, JobStatus.RUNNING),
        ("TERMINATING", None, JobStatus.RUNNING),
        ("TERMINATED", "SUCCESS", JobStatus.COMPLETED),
        ("TERMINATED", "FAILED", JobStatus.FAILED),
        ("TERMINATED", "TIMEDOUT", JobStatus.FAILED),
        ("TERMINATED", "CANCELED", JobStatus.CANCELLED),
        ("INTERNAL_ERROR", None, JobStatus.FAILED),
    ],
)
def test_run_state_translation(life_cycle, result, expected):
    assert map_run_state(life_cycle, result) == expected


def test_partial_success_is_not_reported_as_success():
    # SUCCESS_WITH_FAILURES means at least one task failed; calling that
    # "Completed" would show a user a green run with missing results.
    assert map_run_state("TERMINATED", "SUCCESS_WITH_FAILURES") is JobStatus.FAILED


def test_unknown_states_never_claim_success():
    assert map_run_state("SOME_FUTURE_STATE", None) is JobStatus.RUNNING
    assert map_run_state("TERMINATED", "SOME_FUTURE_RESULT") is JobStatus.FAILED


# ---------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------


def test_submit_stages_dataset_and_config_then_runs_the_job(settings, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)

    run_id = runner.submit(_request(dataset))
    upload_root = f"/Volumes/forecastiq/forecasting/upload_files/runs/{run_id}"
    artifacts_root = f"/Volumes/forecastiq/forecasting/artifacts_files/runs/{run_id}"

    # Only the original dataset lives in uploads.
    assert f"{upload_root}/sales.csv" in workspace.files.uploaded
    assert f"{upload_root}/forecast_configuration.json" not in workspace.files.uploaded
    # Config and every other run artifact live under artifacts instead.
    assert f"{artifacts_root}/forecast_configuration.json" in workspace.files.uploaded

    parameters = _submitted_parameters(workspace)
    assert parameters["dataset"] == f"{upload_root}/sales.csv"
    assert parameters["config"] == f"{artifacts_root}/forecast_configuration.json"
    assert parameters["summary_out"] == f"{artifacts_root}/summary.json"
    assert parameters["live_status_out"] == f"{artifacts_root}/live_status.json"


def test_no_job_parameter_is_ever_empty(settings, dataset):
    """The reason every run value travels in the config file.

    A python_wheel_task's argument list is fixed at deploy time, so an
    empty parameter would reach argparse as `--flag ""` — which crashes on
    --horizon and silently trains a model named "" on --models.
    """
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    runner.submit(_request(dataset))

    for name, value in _submitted_parameters(workspace).items():
        assert value, f"job parameter '{name}' was submitted empty"


def test_config_file_carries_models_fallback_and_horizon(settings, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset))

    root = f"/Volumes/forecastiq/forecasting/artifacts_files/runs/{run_id}"
    payload = json.loads(workspace.files.uploaded[f"{root}/forecast_configuration.json"])

    assert payload["run_id"] == run_id
    assert payload["models"] == ["prophet", "xgboost"]
    assert payload["fallback_model"] == "xgboost"
    assert payload["horizon"] == 18
    assert payload["dataset_name"] == "sales.csv"
    # Multi-valued key columns survive as a real list — the thing a flat
    # job parameter cannot express.
    assert payload["key_columns"] == ["store"]


def test_missing_dataset_fails_the_run_with_a_usable_message(settings, tmp_path):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)

    request = _request(tmp_path / "gone.csv")
    run_id = runner.submit(request)

    # Recorded as FAILED rather than raising: the caller already holds this
    # run id, and a 404 on the next poll would be worse than an honest failure.
    assert runner.get_status(run_id) is JobStatus.FAILED
    assert workspace.jobs.run_now_calls == []


# ---------------------------------------------------------------------
# Status and results
# ---------------------------------------------------------------------


def test_live_stage_trail_is_read_back_from_the_volume(settings, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset))

    root = f"/Volumes/forecastiq/forecasting/artifacts_files/runs/{run_id}"
    workspace.files.uploaded[f"{root}/live_status.json"] = json.dumps(
        {"stages": [{"name": "Load Dataset", "status": "Completed"}]}
    ).encode()

    listing = runner.get_run(run_id)
    assert [stage["name"] for stage in listing.stages] == ["Load Dataset"]


def test_result_is_not_available_while_running(settings, dataset):
    runner = DatabricksRunner(settings, workspace_client=_FakeWorkspace())
    run_id = runner.submit(_request(dataset))

    from app.orchestration.exceptions import RunNotReadyError

    with pytest.raises(RunNotReadyError):
        runner.get_result(run_id)


def test_failure_message_is_translated_not_echoed(settings, dataset):
    state = SimpleNamespace(
        life_cycle_state="TERMINATED",
        result_state="FAILED",
        state_message=(
            "Run failed on https://adb-1234567890123456.4.azuredatabricks.net with "
            "X_SecretResolutionFailure"
        ),
    )
    runner = DatabricksRunner(settings, workspace_client=_FakeWorkspace(state))
    run_id = runner.submit(_request(dataset))

    assert runner.get_status(run_id) is JobStatus.FAILED
    error = runner.get_run(run_id).error
    assert "adb-1234567890123456" not in error
    assert "X_SecretResolutionFailure" not in error
    assert "credential" in error.lower()


def test_cancel_reaches_the_workspace(settings, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset))

    outcome = runner.cancel(run_id)
    assert outcome.cancelled is True
    assert outcome.cleanup_errors == []
    assert workspace.jobs.cancelled == [99]
    # A second cancel has nothing to do and says so rather than pretending.
    assert runner.cancel(run_id).cancelled is False
