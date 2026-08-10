"""DcsRunner — same submit/poll/retrieve flow as DatabricksRunner (see
test_databricks_runner.py for that coverage); these tests exist only for
what genuinely differs: which job it targets and what backend it reports.
"""

import pytest
from types import SimpleNamespace

from app.config.settings import Settings
from app.orchestration.dcs_runner import DcsRunner
from app.orchestration.exceptions import RunNotReadyError
from app.orchestration.schemas import ExecutionBackend, JobStatus, PipelineExecutionRequest


class _FakeFiles:
    def __init__(self) -> None:
        self.uploaded: dict[str, bytes] = {}

    def upload(self, file_path, contents, overwrite=None):
        self.uploaded[file_path] = contents.read()

    def download(self, file_path):
        if file_path not in self.uploaded:
            raise FileNotFoundError(file_path)
        return SimpleNamespace(contents=SimpleNamespace(read=lambda: self.uploaded[file_path]))


class _FakeJobs:
    def __init__(self) -> None:
        self.list_calls: list[str | None] = []
        self.run_now_calls: list[dict] = []

    def list(self, name=None):
        self.list_calls.append(name)
        return [SimpleNamespace(job_id=7777, settings=SimpleNamespace(name=name))]

    def run_now(self, job_id, job_parameters=None):
        self.run_now_calls.append({"job_id": job_id, "job_parameters": job_parameters})
        return SimpleNamespace(run_id=55)

    def get_run(self, run_id, **kwargs):
        return SimpleNamespace(
            state=SimpleNamespace(life_cycle_state="RUNNING", result_state=None, state_message=""),
            run_duration=1000,
        )


class _FakeWorkspace:
    def __init__(self) -> None:
        self.files = _FakeFiles()
        self.jobs = _FakeJobs()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        execution_mode="databricks_dcs",
        databricks_host="https://example.invalid",
        databricks_token="test-token",
        databricks_volumes_root="/Volumes/forecastiq/forecasting/forecast_files",
        upload_dir=str(tmp_path / "uploads"),
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("date,store,sales\n2024-01-01,1,10\n")
    return path


def _request(dataset):
    return PipelineExecutionRequest(
        run_id="dbx-run-dcs-test",
        dataset_path=str(dataset),
        dataset_name="sales.csv",
        forecast_configuration={"date_column": "date", "target_column": "sales"},
    )


def test_dcs_runner_resolves_the_dcs_job_by_its_own_name(settings, dataset):
    workspace = _FakeWorkspace()
    runner = DcsRunner(settings, workspace_client=workspace)

    runner.submit(_request(dataset))

    assert workspace.jobs.list_calls == ["forecastiq-forecast-pipeline-dcs"]
    assert workspace.jobs.run_now_calls[0]["job_id"] == 7777


def test_dcs_runner_reports_the_dcs_backend(settings, dataset):
    workspace = _FakeWorkspace()
    runner = DcsRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset))

    listing = runner.get_run(run_id)
    assert listing.execution_backend is ExecutionBackend.DATABRICKS_DCS

    with pytest.raises(RunNotReadyError):
        runner.get_result(run_id)


def test_dcs_runner_honors_an_explicit_job_id_override(settings, dataset):
    settings = settings.model_copy(update={"databricks_dcs_job_id": 999})
    workspace = _FakeWorkspace()
    runner = DcsRunner(settings, workspace_client=workspace)

    runner.submit(_request(dataset))

    # A pinned id skips the by-name lookup entirely.
    assert workspace.jobs.list_calls == []
    assert workspace.jobs.run_now_calls[0]["job_id"] == 999
