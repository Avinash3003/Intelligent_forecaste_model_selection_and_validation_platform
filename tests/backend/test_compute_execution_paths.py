"""The selected compute is the compute that runs — with no fallback.

Compute Configuration exists so the user decides where the Ray workload
executes. A run must therefore never be redirected to a job resolved by
name, which is how it previously reached the Serverless pipeline.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.orchestration.databricks_runner import _SHARED_JOB_CLUSTER_KEY, TASK_KEYS, DatabricksRunner
from app.orchestration.exceptions import ExecutionError
from app.orchestration.schemas import PipelineExecutionRequest
from app.schemas.compute import ComputeSelection, JobComputeConfig


# Anchored to this file, not the working directory: the source-reading tests
# below must find the same files whether pytest starts at the repo root or in
# backend/.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeFiles:
    def __init__(self) -> None:
        self.uploaded: dict[str, bytes] = {}

    def upload(self, file_path, contents, overwrite=False):
        self.uploaded[file_path] = contents.read()

    def delete_directory(self, path):
        pass


class _FakeJobs:
    """One named Job the runner creates, resets per run, then run_now()s."""

    def __init__(self) -> None:
        self.create_calls: list[dict] = []
        self.reset_calls: list[dict] = []
        self.run_now_calls: list[dict] = []
        self._jobs: dict[int, str] = {}

    def create(self, access_control_list=None, **settings):
        job_id = 4242 + len(self.create_calls)
        self.create_calls.append({**settings, "access_control_list": access_control_list})
        self._jobs[job_id] = settings.get("name")
        return SimpleNamespace(job_id=job_id)

    def list(self, name=None, **kwargs):
        return [
            SimpleNamespace(job_id=jid, settings=SimpleNamespace(name=jname))
            for jid, jname in self._jobs.items()
            if name is None or jname == name
        ]

    def reset(self, job_id=None, new_settings=None):
        self.reset_calls.append({"job_id": job_id, "new_settings": new_settings})

    def run_now(self, job_id=None, job_parameters=None, **kwargs):
        self.run_now_calls.append({"job_id": job_id, "job_parameters": job_parameters})
        return SimpleNamespace(run_id=99)

    def get_run(self, run_id, **kwargs):
        return SimpleNamespace(
            state=SimpleNamespace(life_cycle_state="RUNNING", result_state=None, state_message=""),
            run_duration=1,
            # Databricks builds this itself; the runner never assembles one.
            run_page_url=f"https://example.invalid/#job/4242/run/{run_id}",
        )

    @property
    def submitted_settings(self) -> dict:
        if self.reset_calls:
            new = self.reset_calls[-1]["new_settings"]
            return {f.name: getattr(new, f.name) for f in dataclasses.fields(new)}
        return self.create_calls[-1]


class _FakeWorkspaceFiles:
    """The workspace-files API a DCS run stages through.

    Separate from _FakeFiles on purpose: /Workspace and /Volumes are
    different APIs on the real client, and a test that let one satisfy the
    other would hide a path written by one and read by the other.
    """

    def __init__(self) -> None:
        self.uploaded: dict[str, bytes] = {}
        self.created_dirs: list[str] = []

    def mkdirs(self, path):
        # The real workspace API does not create parents on upload, so the
        # runner calls this first; the fake records it to keep that ordering
        # honest rather than quietly tolerating its absence.
        self.created_dirs.append(path)

    def upload(self, path, content, format=None, overwrite=False):
        self.uploaded[path] = content if isinstance(content, bytes) else content.read()

    def download(self, path):
        import io as _io

        if path not in self.uploaded:
            raise FileNotFoundError(path)
        return _io.BytesIO(self.uploaded[path])


class _FakeCurrentUser:
    def me(self):
        return SimpleNamespace(user_name="sp-forecastiq-cicd")


class _FakeSecrets:
    """Key *names* only, which is all the real API serves — a value is
    never readable, which is why the cluster gets a reference instead."""

    def __init__(self, keys=("azure-openai-endpoint", "azure-openai-api-key", "azure-openai-deployment")):
        self.keys = list(keys)
        self.scopes_listed: list[str] = []

    def list_secrets(self, scope):
        self.scopes_listed.append(scope)
        return [type("S", (), {"key": k})() for k in self.keys]


class _FakeWorkspace:
    def __init__(self, secrets=None) -> None:
        self.files = _FakeFiles()
        self.jobs = _FakeJobs()
        self.current_user = _FakeCurrentUser()
        self.workspace = _FakeWorkspaceFiles()
        self.secrets = secrets or _FakeSecrets()


@pytest.fixture
def settings(tmp_path, mlflow_db):
    return Settings(
        execution_mode="databricks",
        databricks_host="https://example.invalid",
        databricks_token="test-token",
        upload_dir=str(tmp_path / "uploads"),
        mlflow_tracking_uri=f"sqlite:///{mlflow_db}",
    )


@pytest.fixture
def dataset(tmp_path) -> Path:
    path = tmp_path / "sales.csv"
    path.write_text("date,store,sales\n2024-01-01,S1,10\n")
    return path


def _request(dataset: Path, compute) -> PipelineExecutionRequest:
    return PipelineExecutionRequest(
        dataset_path=str(dataset),
        dataset_name="sales.csv",
        forecast_configuration={
            "date_column": "date",
            "target_column": "sales",
            "key_columns": ["store"],
            "feature_columns": [],
        },
        compute=compute,
    )


def _submitted_tasks(workspace):
    return workspace.jobs.submitted_settings["tasks"]


def _submitted_task(workspace):
    """The first DAG task (Load & Prepare). Every task shares the same
    compute attachment, so this is enough for those assertions — the
    cluster spec itself lives on the job's shared job_clusters entry."""
    return _submitted_tasks(workspace)[0]


def _submitted_job_cluster(workspace):
    clusters = workspace.jobs.submitted_settings.get("job_clusters") or []
    assert clusters, "expected one shared job cluster for new_job_compute"
    return clusters[0].new_cluster


def _run(settings, dataset, compute):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset, compute))
    return workspace, runner, run_id


# ---- new job compute ------------------------------------------------


NEW_COMPUTE = ComputeSelection(
    mode="new_job_compute",
    job_compute=JobComputeConfig(
        node_type_id="Standard_E8ads_v7", runtime_key="16.4.x-cpu-ml-scala2.12", num_workers=3
    ),
)


def test_new_compute_reaches_submit_with_the_selected_values(settings, dataset):
    workspace, _, _ = _run(settings, dataset, NEW_COMPUTE)
    cluster = _submitted_job_cluster(workspace)

    assert cluster.node_type_id == "Standard_E8ads_v7"
    assert cluster.spark_version == "16.4.x-cpu-ml-scala2.12"
    assert cluster.num_workers == 3
    assert _submitted_task(workspace).job_cluster_key == _SHARED_JOB_CLUSTER_KEY
    assert _submitted_task(workspace).existing_cluster_id is None


def test_new_compute_carries_autoscale_bounds(settings, dataset):
    compute = ComputeSelection(
        mode="new_job_compute",
        job_compute=JobComputeConfig(
            node_type_id="Standard_F4ads_v7",
            runtime_key="15.4.x-cpu-ml-scala2.12",
            autoscale=True,
            min_workers=2,
            max_workers=7,
        ),
    )
    workspace, _, _ = _run(settings, dataset, compute)
    cluster = _submitted_job_cluster(workspace)

    assert cluster.autoscale.min_workers == 2
    assert cluster.autoscale.max_workers == 7
    assert cluster.num_workers is None


def test_single_node_new_compute_is_marked_single_node(settings, dataset):
    compute = ComputeSelection(
        mode="new_job_compute",
        job_compute=JobComputeConfig(
            node_type_id="Standard_DC4as_v5", runtime_key="15.4.x-cpu-ml-scala2.12", num_workers=0
        ),
    )
    workspace, _, _ = _run(settings, dataset, compute)
    cluster = _submitted_job_cluster(workspace)

    assert cluster.num_workers == 0
    assert cluster.custom_tags["ResourceClass"] == "SingleNode"


# ---- Databricks Container Services (docker_image) --------------------
#
# A new job cluster pulls the configured runtime image instead of resolving
# its dependencies from the Databricks runtime, when Container Services is
# configured. `databricks_docker_image_url` is the single on/off switch --
# blank means DCS stays off and the cluster is built exactly as it always
# was, matching every other optional Databricks feature on Settings.


def test_docker_image_is_absent_when_dcs_is_not_configured(settings, dataset):
    """Configuration loading: the default Settings has no DCS fields set,
    so a new job cluster must come back with no docker_image at all --
    this must never require an explicit opt-out."""
    workspace, _, _ = _run(settings, dataset, NEW_COMPUTE)
    cluster = _submitted_job_cluster(workspace)

    assert cluster.docker_image is None


def test_new_compute_attaches_the_configured_docker_image(settings, dataset):
    dcs_settings = settings.model_copy(
        update={
            "databricks_docker_image_url": "avinashforecastiqacr.azurecr.io/forecastiq-runtime:v1",
            "databricks_docker_image_username": "sp-forecastiq-dcs-acrpull",
            "databricks_docker_image_password": "super-secret-value",
        }
    )
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(dcs_settings, workspace_client=workspace)
    runner.submit(_request(dataset, NEW_COMPUTE))
    cluster = _submitted_job_cluster(workspace)

    assert cluster.docker_image.url == "avinashforecastiqacr.azurecr.io/forecastiq-runtime:v1"
    assert cluster.docker_image.basic_auth.username == "sp-forecastiq-dcs-acrpull"
    assert cluster.docker_image.basic_auth.password == "super-secret-value"
    # Every other field is untouched -- DCS is additive, not a rewrite.
    assert cluster.node_type_id == "Standard_E8ads_v7"
    assert cluster.num_workers == 3


def test_dcs_downgrades_the_runtime_to_its_standard_non_ml_equivalent(settings, dataset):
    """Pairing an ML runtime with a Docker image is self-contradictory: the
    image supplies the Python/dependency stack INSTEAD OF the ML runtime's
    own. docker_image alone is not enough -- the version string must say so
    too.

    use_ml_runtime is deliberately never set here -- verified against a real
    jobs.submit call, which rejects the field outright when set explicitly
    without a `kind` ("use_ml_runtime is not allowed with unspecified
    kind"). Leaving it unset and downgrading spark_version alone submits
    cleanly; Databricks infers the flag from the version string."""
    dcs_settings = settings.model_copy(
        update={"databricks_docker_image_url": "avinashforecastiqacr.azurecr.io/forecastiq-runtime:v1"}
    )
    workspace = _FakeWorkspace()
    DatabricksRunner(dcs_settings, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))
    cluster = _submitted_job_cluster(workspace)

    # NEW_COMPUTE selects "16.4.x-cpu-ml-scala2.12" -- an ML runtime preset.
    assert cluster.spark_version == "16.4.x-scala2.12"
    assert cluster.use_ml_runtime is None


def test_dcs_sets_the_single_user_name_for_the_job_triggered_cluster(settings, dataset):
    """A SINGLE_USER cluster this backend creates for one run should be
    scoped to the identity actually running it -- the same service
    principal every other Databricks call in this process authenticates
    as, resolved the same way compute_service's own probe already does."""
    dcs_settings = settings.model_copy(
        update={"databricks_docker_image_url": "avinashforecastiqacr.azurecr.io/forecastiq-runtime:v1"}
    )
    workspace = _FakeWorkspace()
    DatabricksRunner(dcs_settings, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))
    cluster = _submitted_job_cluster(workspace)

    assert cluster.single_user_name == "sp-forecastiq-cicd"


def test_without_dcs_the_runtime_and_ml_flag_are_left_alone(settings, dataset):
    """The fix only applies when a Docker image is actually attached --
    the plain ML-runtime path (DCS off) must render byte-identical to
    before this change."""
    workspace = _FakeWorkspace()
    DatabricksRunner(settings, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))
    cluster = _submitted_job_cluster(workspace)

    assert cluster.spark_version == "16.4.x-cpu-ml-scala2.12"
    assert cluster.use_ml_runtime is None
    assert cluster.single_user_name is None


def test_a_dcs_run_stages_straight_into_unity_catalog(settings, dataset):
    """A container has no `/Volumes` POSIX mount, but that was never a lack
    of access -- only of a filesystem handler.

    Proven on the production image (wheel-task run 130735570011315):
    `os.listdir("/Volumes")` raises `PermissionError [Errno 1]`, while the
    Files API lists, writes and reads back byte-identical. So a DCS run
    stages to the UC Volume like any other run, and the storage adapter
    reaches it over the API. Nothing is copied into the workspace, which is
    what used to put raw datasets and model binaries under a folder the
    `users` group could manage."""
    dcs_settings = settings.model_copy(
        update={"databricks_docker_image_url": "acr.example.io/forecastiq-runtime:v1"}
    )
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(dcs_settings, workspace_client=workspace)
    runner.submit(_request(dataset, NEW_COMPUTE))

    staged = list(workspace.files.uploaded)
    assert staged, "a DCS run must stage to the UC Volume"
    assert all(p.startswith("/Volumes/") for p in staged), staged
    assert not list(workspace.workspace.uploaded), "nothing may be staged in the workspace"


def test_a_non_dcs_run_still_stages_to_uc_volumes(settings, dataset):
    """The existing-compute path is unchanged: no docker image, so UC
    Volumes remain the staging location they have always been."""
    workspace, _, _ = _run(settings, dataset, NEW_COMPUTE)

    staged = list(workspace.files.uploaded)
    assert staged, "a non-DCS run must stage through the UC Volume API"
    assert all(p.startswith("/Volumes/") for p in staged), staged
    assert workspace.workspace.uploaded == {}


def test_the_engine_is_pointed_at_the_same_paths_it_was_given(settings, dataset):
    """Whatever root a run stages to, the engine must be told that root --
    a run staged to /Workspace but told to read /Volumes would fail exactly
    the way the original bug did."""
    dcs_settings = settings.model_copy(
        update={"databricks_docker_image_url": "acr.example.io/forecastiq-runtime:v1"}
    )
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(dcs_settings, workspace_client=workspace)
    runner.submit(_request(dataset, NEW_COMPUTE))

    supplied = workspace.jobs.run_now_calls[-1]["job_parameters"]
    for name in ("dataset", "config", "summary_out", "live_status_out"):
        assert supplied[name].startswith("/Volumes/"), f"{name} -> {supplied[name]}"


def test_docker_image_url_is_never_hardcoded_in_python(settings, dataset):
    """Two different configured URLs must produce two different clusters --
    proof the value comes from settings, not a literal in the runner."""
    first_settings = settings.model_copy(
        update={"databricks_docker_image_url": "avinashforecastiqacr.azurecr.io/forecastiq-runtime:v1"}
    )
    first_ws = _FakeWorkspace()
    DatabricksRunner(first_settings, workspace_client=first_ws).submit(_request(dataset, NEW_COMPUTE))

    second_settings = settings.model_copy(
        update={"databricks_docker_image_url": "otheracr.azurecr.io/some-other-image:v9"}
    )
    second_ws = _FakeWorkspace()
    DatabricksRunner(second_settings, workspace_client=second_ws).submit(_request(dataset, NEW_COMPUTE))

    assert _submitted_job_cluster(first_ws).docker_image.url.endswith("forecastiq-runtime:v1")
    assert _submitted_job_cluster(second_ws).docker_image.url.endswith("some-other-image:v9")


def test_docker_image_without_credentials_has_no_basic_auth(settings, dataset):
    """A public or already-cached image needs no credential -- basic_auth
    must not be forced onto a spec that never asked for it."""
    dcs_settings = settings.model_copy(
        update={"databricks_docker_image_url": "avinashforecastiqacr.azurecr.io/forecastiq-runtime:v1"}
    )
    workspace = _FakeWorkspace()
    DatabricksRunner(dcs_settings, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))
    cluster = _submitted_job_cluster(workspace)

    assert cluster.docker_image.url
    assert cluster.docker_image.basic_auth is None


def test_existing_compute_path_never_touches_docker_image(settings, dataset):
    """Existing compute is preserved exactly as it was: it attaches by
    cluster id and must never gain a docker_image, even when DCS is
    configured for new job compute."""
    dcs_settings = settings.model_copy(
        update={
            "databricks_docker_image_url": "avinashforecastiqacr.azurecr.io/forecastiq-runtime:v1",
            "databricks_docker_image_username": "sp-forecastiq-dcs-acrpull",
            "databricks_docker_image_password": "super-secret-value",
        }
    )
    existing = ComputeSelection(mode="existing_compute", cluster_id="0826-092202-g7bkkhgi")
    workspace = _FakeWorkspace()
    DatabricksRunner(dcs_settings, workspace_client=workspace).submit(_request(dataset, existing))
    task = _submitted_task(workspace)

    assert task.existing_cluster_id == "0826-092202-g7bkkhgi"
    assert task.new_cluster is None


def test_the_acr_password_never_reaches_a_run_id_or_error_message(settings, dataset, caplog):
    """A submission failure must still redact the password -- the same
    guarantee every other Databricks credential on this class already has."""
    dcs_settings = settings.model_copy(
        update={
            "databricks_docker_image_url": "avinashforecastiqacr.azurecr.io/forecastiq-runtime:v1",
            "databricks_docker_image_username": "sp-forecastiq-dcs-acrpull",
            "databricks_docker_image_password": "super-secret-value",
        }
    )

    class _FailingJobs(_FakeJobs):
        def submit(self, run_name=None, tasks=None, access_control_list=None):
            raise RuntimeError(f"denied for password=super-secret-value on {run_name}")

    workspace = _FakeWorkspace()
    workspace.jobs = _FailingJobs()
    runner = DatabricksRunner(dcs_settings, workspace_client=workspace)

    run_id = runner.submit(_request(dataset, NEW_COMPUTE))
    status = runner.get_run(run_id)

    assert "super-secret-value" not in (status.error or "")
    assert "super-secret-value" not in caplog.text


def test_dcs_runs_the_job_this_runner_defined_itself(settings, dataset):
    """The old Serverless design ran a job someone else had deployed, whose
    task spec this app did not control. The job run here is one this runner
    wrote in the same call, and DCS goes through that same path as every
    other compute selection rather than a separate one."""
    dcs_settings = settings.model_copy(
        update={"databricks_docker_image_url": "avinashforecastiqacr.azurecr.io/forecastiq-runtime:v1"}
    )
    workspace = _FakeWorkspace()
    DatabricksRunner(dcs_settings, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))

    assert len(workspace.jobs.run_now_calls) == 1
    assert [task.task_key for task in _submitted_tasks(workspace)] == list(TASK_KEYS)


def test_no_node_type_or_worker_count_is_hardcoded(settings, dataset):
    """Two different selections must produce two different clusters."""
    first, _, _ = _run(settings, dataset, NEW_COMPUTE)
    second_compute = ComputeSelection(
        mode="new_job_compute",
        job_compute=JobComputeConfig(
            node_type_id="Standard_L4aos_v4", runtime_key="15.4.x-cpu-ml-scala2.12", num_workers=1
        ),
    )
    second, _, _ = _run(settings, dataset, second_compute)

    assert _submitted_job_cluster(first).node_type_id != _submitted_job_cluster(second).node_type_id
    assert _submitted_job_cluster(first).num_workers != _submitted_job_cluster(second).num_workers


# ---- existing compute -----------------------------------------------


EXISTING = ComputeSelection(mode="existing_compute", cluster_id="0826-abc-chosen")


def test_existing_compute_runs_on_the_selected_cluster(settings, dataset):
    workspace, _, _ = _run(settings, dataset, EXISTING)
    task = _submitted_task(workspace)

    assert task.existing_cluster_id == "0826-abc-chosen"
    assert task.new_cluster is None


def test_where_a_run_executes_comes_from_the_selection_not_the_job(settings, dataset):
    """The job is reused across runs, but its compute is rewritten from the
    selection before each one — a name lookup never decides where a run lands."""
    workspace, _, _ = _run(settings, dataset, EXISTING)

    assert _submitted_task(workspace).existing_cluster_id == "0826-abc-chosen"
    assert not hasattr(settings, "databricks_job_name")


def test_both_modes_submit_the_same_wheel_task(settings, dataset):
    new_task = _submitted_task(_run(settings, dataset, NEW_COMPUTE)[0])
    existing_task = _submitted_task(_run(settings, dataset, EXISTING)[0])

    assert new_task.python_wheel_task.package_name == existing_task.python_wheel_task.package_name
    assert new_task.python_wheel_task.entry_point == existing_task.python_wheel_task.entry_point
    # Run ids differ, so compare the argument shape rather than the paths.
    flags = lambda task: [a for a in task.python_wheel_task.parameters if a.startswith("--")]
    assert flags(new_task) == flags(existing_task)
    assert "--parallel-keys" in flags(new_task)
    assert "--stage" in flags(new_task)


def test_every_task_key_is_present_and_wired_in_dependency_order(settings, dataset):
    """The real Databricks DAG: seven tasks, each depending only on the one
    directly before it — this is what makes the Jobs UI render a workflow
    graph instead of a single node."""
    workspace, _, _ = _run(settings, dataset, NEW_COMPUTE)
    tasks = _submitted_tasks(workspace)

    assert [task.task_key for task in tasks] == list(TASK_KEYS)
    assert tasks[0].depends_on is None
    for task, previous_key in zip(tasks[1:], TASK_KEYS):
        assert [dep.task_key for dep in task.depends_on] == [previous_key]
    for task in tasks:
        # By flag, not by position: the parameter list also carries the
        # credential references, which are appended after this pair.
        params = task.python_wheel_task.parameters
        assert params[params.index("--stage") + 1] == task.task_key


def test_ray_stages_share_one_job_cluster_not_one_per_task(settings, dataset):
    """Ray parallelism lives inside train/evaluate/explain/rank_select's own
    tasks; the cluster those tasks run on must still be booted once for the
    whole run, not once per task. jobs.submit() (a one-time run) has no
    job_clusters/job_cluster_key support in the installed SDK, so this is
    done by creating one real cluster up front and pointing every task at
    it by existing_cluster_id."""
    workspace, _, _ = _run(settings, dataset, NEW_COMPUTE)

    assert len(workspace.jobs.submitted_settings["job_clusters"]) == 1
    for task in _submitted_tasks(workspace):
        assert task.job_cluster_key == _SHARED_JOB_CLUSTER_KEY
        assert task.new_cluster is None


def test_the_cluster_carries_this_backend_s_mlflow_settings(settings, dataset):
    """A DCS image replaces the Databricks runtime's own environment, which
    is what normally pre-sets MLFLOW_TRACKING_URI. Without forwarding it the
    engine falls back to a local sqlite file inside the container and the run
    never reaches the history this backend reads back."""
    tracked = settings.model_copy(
        update={"mlflow_tracking_uri": "databricks", "mlflow_experiment_name": "/forecast-engine"}
    )
    workspace = _FakeWorkspace()
    DatabricksRunner(tracked, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))

    env = _submitted_job_cluster(workspace).spark_env_vars
    assert env["MLFLOW_TRACKING_URI"] == "databricks"
    assert env["MLFLOW_EXPERIMENT_NAME"] == "/forecast-engine"


def test_the_cluster_tracks_where_this_backend_reads_when_nothing_is_configured(settings, dataset):
    """The symptom this pins: a finished run that never appears in history.

    A normal deployment leaves MLFLOW_TRACKING_URI unset. The backend
    resolves that to "databricks" for this execution mode; the cluster must
    be given the same answer, or the engine writes to MLflowConfig's local
    sqlite default inside the container while the backend reads the
    workspace store."""
    unset = settings.model_copy(update={"mlflow_tracking_uri": None, "mlflow_experiment_name": None})
    workspace = _FakeWorkspace()
    DatabricksRunner(unset, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))

    env = _submitted_job_cluster(workspace).spark_env_vars
    assert env["MLFLOW_TRACKING_URI"] == unset.mlflow_tracking_uri_resolved == "databricks"
    assert env["MLFLOW_EXPERIMENT_NAME"] == unset.mlflow_experiment_name_resolved


def test_a_setting_with_no_resolver_is_omitted_rather_than_sent_empty(settings, dataset):
    """Tracking URI and experiment name both resolve to a real default, so
    they are always sent. The registry URI has no resolver — unset means
    unset, and an empty string would be a value the engine then honours."""
    untracked = settings.model_copy(
        update={"mlflow_tracking_uri": None, "mlflow_registry_uri": None, "mlflow_experiment_name": None}
    )
    workspace = _FakeWorkspace()
    DatabricksRunner(untracked, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))

    assert "MLFLOW_REGISTRY_URI" not in _submitted_job_cluster(workspace).spark_env_vars


def test_no_all_purpose_cluster_is_ever_created_for_a_run(settings, dataset):
    """A job cluster belongs to the run and Databricks disposes of it. The
    earlier design created an ordinary all-purpose cluster per run instead,
    which billed at the higher rate and piled up in the Compute list."""
    workspace, runner, run_id = _run(settings, dataset, NEW_COMPUTE)

    assert not hasattr(workspace, "clusters") or not getattr(workspace.clusters, "create_calls", [])
    assert workspace.jobs.submitted_settings["job_clusters"][0].job_cluster_key == _SHARED_JOB_CLUSTER_KEY


def test_every_run_is_a_run_of_the_same_named_job(settings, dataset):
    """All runs belong to one job, so its run history is the whole forecast
    history — a second run reuses the job rather than defining another."""
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    runner.submit(_request(dataset, NEW_COMPUTE))
    runner.submit(_request(dataset, NEW_COMPUTE))

    assert len(workspace.jobs.create_calls) == 1
    assert len(workspace.jobs.run_now_calls) == 2
    assert {call["job_id"] for call in workspace.jobs.run_now_calls} == {4242}
    assert workspace.jobs.create_calls[0]["name"] == settings.databricks_job_display_name


# ---- safety ----------------------------------------------------------


@pytest.mark.parametrize(
    "compute, expected",
    [
        (None, "Select the compute"),
        (ComputeSelection(mode="existing_compute", cluster_id=None), "missing its cluster id"),
        (ComputeSelection(mode="existing_compute", cluster_id="   "), "missing its cluster id"),
    ],
)
def test_missing_compute_fails_clearly(settings, dataset, compute, expected):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset, compute))

    # Recorded as a failed run rather than raised; the caller holds the id.
    from app.orchestration.schemas import JobStatus

    assert runner.get_status(run_id) is JobStatus.FAILED
    assert workspace.jobs.run_now_calls == []


def test_unknown_compute_mode_is_rejected():
    with pytest.raises(ValueError):
        ComputeSelection(mode="serverless")


# Resolving a job someone ELSE deployed, by a configured name or id, is how
# execution previously reached the Serverless pipeline. `run_now` itself is
# no longer forbidden — this runner calls it on the job it defines and owns
# — but a configured foreign job name/id must never come back.
FORBIDDEN_ROUTING = ("_resolve_job_id", "databricks_job_name", "databricks_job_id")


def test_no_application_module_routes_a_run_to_a_foreign_job():
    offenders = []
    for path in (_REPO_ROOT / "backend" / "app").rglob("*.py"):
        source = path.read_text()
        for token in FORBIDDEN_ROUTING:
            if token in source:
                offenders.append(f"{path}: {token}")
    assert not offenders, f"Serverless routing reachable from: {offenders}"


def test_the_engine_is_never_asked_to_run_a_serverless_stage_group():
    """The real DAG (TASK_KEYS, --stage) is not the old Serverless design
    this guards against: no per-key stage-group flag, and no ad hoc
    checkpoint-dir CLI flag — the checkpoint handoff lives entirely inside
    forecast_engine/core/checkpoint.py, resolved from the run's own
    persisted artifacts config, never a path passed on the command line."""
    settings_source = (
        _REPO_ROOT / "backend" / "app" / "orchestration" / "databricks_runner.py"
    ).read_text()
    assert "--stage-group" not in settings_source
    assert "--checkpoint-dir" not in settings_source


# ---- regression ------------------------------------------------------


def test_volume_paths_and_config_staging_are_unchanged(settings, dataset):
    workspace, _, run_id = _run(settings, dataset, EXISTING)
    upload_root = f"{settings.databricks_uploads_volumes_root}/runs/{run_id}"
    artifacts_root = f"{settings.databricks_artifacts_volumes_root}/runs/{run_id}"

    assert f"{upload_root}/sales.csv" in workspace.files.uploaded
    config = json.loads(workspace.files.uploaded[f"{artifacts_root}/forecast_configuration.json"])
    assert config["date_column"] == "date"
    assert config["curated_storage"]["root_dir"].startswith(settings.databricks_curated_volumes_root)

    supplied = workspace.jobs.run_now_calls[-1]["job_parameters"]
    assert supplied["summary_out"] == f"{artifacts_root}/summary.json"
    assert supplied["live_status_out"] == f"{artifacts_root}/live_status.json"


# --- Azure OpenAI credentials reach the cluster ------------------------
#
# Insights came back with provider "template" and error "AZURE_OPENAI_
# ENDPOINT is not set" on every Databricks run, while the same credentials
# worked locally: Settings.subprocess_env forwards them to the local engine
# subprocess, and this path forwarded only the MLFLOW_* ones.
#
# They travel as `{{secrets/<scope>/<key>}}`, never as values. A job
# definition's spark_env_vars are readable by anyone who can view the job,
# so a plaintext API key there is a published key.


def _task_args(workspace, task_index=0):
    return _submitted_tasks(workspace)[task_index].python_wheel_task.parameters


def _flag_value(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


@pytest.mark.parametrize("compute", [NEW_COMPUTE, EXISTING], ids=["new_job", "existing"])
def test_both_compute_modes_are_told_the_same_secret_scope(settings, dataset, compute):
    # One path, no drift: an existing cluster's environment was fixed by
    # whoever created it, so cluster env vars reach only one of these two.
    workspace = _FakeWorkspace()
    DatabricksRunner(settings, workspace_client=workspace).submit(_request(dataset, compute))

    args = _task_args(workspace)

    assert _flag_value(args, "--databricks-secret-scope") == "forecastiq"


def test_every_task_is_told_the_scope_not_just_the_first(settings, dataset):
    # Each phase is its own process, and business_insights runs in the last task.
    workspace = _FakeWorkspace()
    DatabricksRunner(settings, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))

    for index in range(len(TASK_KEYS)):
        assert _flag_value(_task_args(workspace, index), "--databricks-secret-scope") == "forecastiq"


def test_no_credential_and_no_reference_is_sent_at_all(settings, dataset):
    # The backend sends a scope name. Not a value, and not a {{secrets/...}}
    # reference either — the engine names the keys it needs.
    configured = settings.model_copy(
        update={
            "azure_openai_api_key": "fake-key-value-aaaa",
            "azure_openai_endpoint": "https://fake.openai.azure.com/",
        }
    )
    workspace = _FakeWorkspace()
    DatabricksRunner(configured, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))

    definition = repr(workspace.jobs.submitted_settings)

    assert "fake-key-value-aaaa" not in definition
    assert "https://fake.openai.azure.com/" not in definition
    assert "{{secrets/" not in definition


def test_the_backend_never_calls_the_secrets_api(settings, dataset):
    # Nothing here reads key names or values; that is the simplification.
    workspace = _FakeWorkspace()
    DatabricksRunner(settings, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))

    assert workspace.secrets.scopes_listed == []


def test_no_credential_is_placed_in_the_cluster_environment(settings, dataset):
    workspace = _FakeWorkspace()
    DatabricksRunner(settings, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))

    env = _submitted_job_cluster(workspace).spark_env_vars

    assert "AZURE_OPENAI_API_KEY" not in env
    assert "AZURE_OPENAI_ENDPOINT" not in env
    assert "AZURE_OPENAI_DEPLOYMENT_NAME" not in env
    # Non-secret settings still travel this way for a cluster we create.
    assert env["MLFLOW_TRACKING_URI"] == settings.mlflow_tracking_uri_resolved


def test_no_scope_configured_sends_no_flag(settings, dataset):
    workspace = _FakeWorkspace()
    unscoped = settings.model_copy(update={"databricks_secret_scope": ""})
    DatabricksRunner(unscoped, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))

    assert _flag_value(_task_args(workspace), "--databricks-secret-scope") is None


def test_no_all_purpose_cluster_is_ever_created_for_a_run(settings, dataset):
    """A job cluster belongs to the run and Databricks disposes of it. The
    earlier design created an ordinary all-purpose cluster per run instead,
    which billed at the higher rate and piled up in the Compute list."""
    workspace, runner, run_id = _run(settings, dataset, NEW_COMPUTE)

    assert not hasattr(workspace, "clusters") or not getattr(workspace.clusters, "create_calls", [])
    assert workspace.jobs.submitted_settings["job_clusters"][0].job_cluster_key == _SHARED_JOB_CLUSTER_KEY


def test_every_run_is_a_run_of_the_same_named_job(settings, dataset):
    """All runs belong to one job, so its run history is the whole forecast
    history — a second run reuses the job rather than defining another."""
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    runner.submit(_request(dataset, NEW_COMPUTE))
    runner.submit(_request(dataset, NEW_COMPUTE))

    assert len(workspace.jobs.create_calls) == 1
    assert len(workspace.jobs.run_now_calls) == 2
    assert {call["job_id"] for call in workspace.jobs.run_now_calls} == {4242}
    assert workspace.jobs.create_calls[0]["name"] == settings.databricks_job_display_name


# ---- safety ----------------------------------------------------------


@pytest.mark.parametrize(
    "compute, expected",
    [
        (None, "Select the compute"),
        (ComputeSelection(mode="existing_compute", cluster_id=None), "missing its cluster id"),
        (ComputeSelection(mode="existing_compute", cluster_id="   "), "missing its cluster id"),
    ],
)
def test_missing_compute_fails_clearly(settings, dataset, compute, expected):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset, compute))

    # Recorded as a failed run rather than raised; the caller holds the id.
    from app.orchestration.schemas import JobStatus

    assert runner.get_status(run_id) is JobStatus.FAILED
    assert workspace.jobs.run_now_calls == []


def test_unknown_compute_mode_is_rejected():
    with pytest.raises(ValueError):
        ComputeSelection(mode="serverless")


# Resolving a job someone ELSE deployed, by a configured name or id, is how
# execution previously reached the Serverless pipeline. `run_now` itself is
# no longer forbidden — this runner calls it on the job it defines and owns
# — but a configured foreign job name/id must never come back.
FORBIDDEN_ROUTING = ("_resolve_job_id", "databricks_job_name", "databricks_job_id")


def test_no_application_module_routes_a_run_to_a_foreign_job():
    offenders = []
    for path in (_REPO_ROOT / "backend" / "app").rglob("*.py"):
        source = path.read_text()
        for token in FORBIDDEN_ROUTING:
            if token in source:
                offenders.append(f"{path}: {token}")
    assert not offenders, f"Serverless routing reachable from: {offenders}"


def test_the_engine_is_never_asked_to_run_a_serverless_stage_group():
    """The real DAG (TASK_KEYS, --stage) is not the old Serverless design
    this guards against: no per-key stage-group flag, and no ad hoc
    checkpoint-dir CLI flag — the checkpoint handoff lives entirely inside
    forecast_engine/core/checkpoint.py, resolved from the run's own
    persisted artifacts config, never a path passed on the command line."""
    settings_source = (
        _REPO_ROOT / "backend" / "app" / "orchestration" / "databricks_runner.py"
    ).read_text()
    assert "--stage-group" not in settings_source
    assert "--checkpoint-dir" not in settings_source


# ---- regression ------------------------------------------------------


def test_volume_paths_and_config_staging_are_unchanged(settings, dataset):
    workspace, _, run_id = _run(settings, dataset, EXISTING)
    upload_root = f"{settings.databricks_uploads_volumes_root}/runs/{run_id}"
    artifacts_root = f"{settings.databricks_artifacts_volumes_root}/runs/{run_id}"

    assert f"{upload_root}/sales.csv" in workspace.files.uploaded
    config = json.loads(workspace.files.uploaded[f"{artifacts_root}/forecast_configuration.json"])
    assert config["date_column"] == "date"
    assert config["curated_storage"]["root_dir"].startswith(settings.databricks_curated_volumes_root)

    supplied = workspace.jobs.run_now_calls[-1]["job_parameters"]
    assert supplied["summary_out"] == f"{artifacts_root}/summary.json"
    assert supplied["live_status_out"] == f"{artifacts_root}/live_status.json"


# --- Azure OpenAI credentials reach the cluster ------------------------
#
# Insights came back with provider "template" and error "AZURE_OPENAI_
# ENDPOINT is not set" on every Databricks run, while the same credentials
# worked locally: Settings.subprocess_env forwards them to the local engine
# subprocess, and this path forwarded only the MLFLOW_* ones.
#
# They travel as `{{secrets/<scope>/<key>}}`, never as values. A job
# definition's spark_env_vars are readable by anyone who can view the job,
# so a plaintext API key there is a published key.


def _task_args(workspace, task_index=0):
    return _submitted_tasks(workspace)[task_index].python_wheel_task.parameters


def _flag_value(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def test_no_credential_is_placed_in_the_cluster_environment(settings, dataset):
    """Cluster env vars are the path that cannot serve Existing Compute, so
    no secret may travel that way at all."""
    workspace = _FakeWorkspace()
    DatabricksRunner(settings, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))

    env = _submitted_job_cluster(workspace).spark_env_vars

    assert "AZURE_OPENAI_API_KEY" not in env
    assert "AZURE_OPENAI_ENDPOINT" not in env
    assert "AZURE_OPENAI_DEPLOYMENT_NAME" not in env
    # Non-secret settings still travel this way for a cluster we create.
    assert env["MLFLOW_TRACKING_URI"] == settings.mlflow_tracking_uri_resolved


def test_an_unreadable_scope_does_not_stop_the_run(settings, dataset):
    class _Exploding:
        def list_secrets(self, scope):
            raise RuntimeError("PERMISSION_DENIED")

    workspace = _FakeWorkspace(secrets=_Exploding())
    DatabricksRunner(settings, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))

    args = _task_args(workspace)

    assert _flag_value(args, "--azure-openai-api-key") is None
    assert "--dataset" in args  # the run itself is unaffected


def test_no_scope_configured_sends_no_credentials(settings, dataset):
    workspace = _FakeWorkspace()
    unscoped = settings.model_copy(update={"databricks_secret_scope": ""})
    DatabricksRunner(unscoped, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))

    assert _flag_value(_task_args(workspace), "--azure-openai-api-key") is None
    assert workspace.secrets.scopes_listed == []


# --- the suite must not depend on where it was started ----------------
#
# `Settings` resolves env_file=".env" against the working directory, and
# several tests read source files by relative path. From backend/ the first
# loaded the developer's real credentials and 14 tests failed on values no
# test set; from forecast_engine/ the second silently *skipped* three
# checks, one of them "the Dockerfile mentions no secrets". Both directions
# are dangerous: real config can satisfy a test whose whole purpose is to
# prove the code supplies that config.


# Touching the filesystem through a literal path, on the same line or via a
# module constant read later. A Path() built only to be passed as an argument
# never resolves against the disk, so it is not a finding.
_READS_DISK = (".read_text(", ".rglob(", ".glob(", ".exists(", ".open(", ".iterdir(", ".is_file(")


def test_no_test_reads_a_source_file_by_a_working_directory_relative_path():
    offenders = []
    for path in (_REPO_ROOT / "tests").rglob("test_*.py"):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or 'Path("' not in stripped or "__file__" in stripped:
                continue
            module_constant = line[:1].isupper() or line[:1] == "_"
            if module_constant or any(marker in stripped for marker in _READS_DISK):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{number}: {stripped}")

    assert not offenders, "anchor these to __file__ instead:\n" + "\n".join(offenders)


def test_settings_in_the_suite_never_read_a_developer_env_file():
    """tests/conftest.py pins this; without it the answer depends on cwd."""
    assert Settings.model_config["env_file"] is None


# --- the live run is reachable while it is still live -----------------
#
# The link to the Databricks run existed only after the run appeared in
# history, and the submission confirmation told the user there was "no need
# to open Databricks". That inverts the feature: watching stages execute is
# only possible *during* execution. The run id the link is built from is
# known the moment the job is triggered, so the URL ships with the
# submission response.


def test_the_run_page_url_is_known_as_soon_as_the_job_is_triggered(settings, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset, NEW_COMPUTE))

    listing = runner.get_run(run_id)

    assert listing.databricks_run_url == "https://example.invalid/#job/4242/run/99"


def test_the_url_survives_a_workspace_that_offers_none(settings, dataset):
    """A run that submitted fine is never failed by a missing link."""

    class _NoUrlJobs(_FakeJobs):
        def get_run(self, run_id, **kwargs):
            got = super().get_run(run_id, **kwargs)
            got.run_page_url = None
            return got

    workspace = _FakeWorkspace()
    workspace.jobs = _NoUrlJobs()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset, NEW_COMPUTE))

    assert runner.get_run(run_id).databricks_run_url is None


# --- credentials must not be reachable from anything a person can read ---


