"""The selected compute is the compute that runs — with no fallback.

Compute Configuration exists so the user decides where the Ray workload
executes. A run must therefore never be redirected to a job resolved by
name, which is how it previously reached the Serverless pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.orchestration.databricks_runner import TASK_KEYS, DatabricksRunner
from app.orchestration.exceptions import ExecutionError
from app.orchestration.schemas import PipelineExecutionRequest
from app.schemas.compute import ComputeSelection, JobComputeConfig


class _FakeFiles:
    def __init__(self) -> None:
        self.uploaded: dict[str, bytes] = {}

    def upload(self, file_path, contents, overwrite=False):
        self.uploaded[file_path] = contents.read()

    def delete_directory(self, path):
        pass


class _FakeJobs:
    def __init__(self) -> None:
        self.submit_calls: list[dict] = []

    def submit(self, run_name=None, tasks=None, access_control_list=None):
        self.submit_calls.append(
            {"run_name": run_name, "tasks": tasks, "access_control_list": access_control_list}
        )
        return SimpleNamespace(run_id=99)

    def get_run(self, run_id, **kwargs):
        return SimpleNamespace(
            state=SimpleNamespace(life_cycle_state="RUNNING", result_state=None, state_message=""),
            run_duration=1,
        )


class _FakeClusters:
    """Stands in for the real Clusters API — jobs.submit() (a one-time run)
    has no job_clusters/job_cluster_key support in the installed SDK, so a
    new_job_compute run creates one real cluster here and every task
    attaches to it by existing_cluster_id (see _create_shared_cluster)."""

    def __init__(self) -> None:
        self.create_calls: list[dict] = []
        self.created_ids: list[str] = []
        self.deleted_ids: list[str] = []

    def create(self, **kwargs):
        cluster_id = f"managed-cluster-{len(self.create_calls)}"
        self.create_calls.append(kwargs)
        self.created_ids.append(cluster_id)
        details = SimpleNamespace(cluster_id=cluster_id)
        return SimpleNamespace(response=details, result=lambda timeout=None: details)

    def delete(self, cluster_id=None):
        self.deleted_ids.append(cluster_id)


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


class _FakeWorkspace:
    def __init__(self) -> None:
        self.files = _FakeFiles()
        self.jobs = _FakeJobs()
        self.clusters = _FakeClusters()
        self.current_user = _FakeCurrentUser()
        self.workspace = _FakeWorkspaceFiles()


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
    return workspace.jobs.submit_calls[0]["tasks"]


def _submitted_task(workspace):
    """The first DAG task (Load & Prepare). Every task in a run shares the
    same existing_cluster_id, so this is enough for compute-attachment
    assertions — the cluster spec itself now lives in what was passed to
    clusters.create() (_submitted_job_cluster), not on any one task."""
    return _submitted_tasks(workspace)[0]


def _submitted_job_cluster(workspace):
    calls = workspace.clusters.create_calls
    assert calls, "expected one shared cluster to be created for new_job_compute"
    return SimpleNamespace(**calls[0])


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
    created_id = workspace.clusters.created_ids[0]
    assert _submitted_task(workspace).existing_cluster_id == created_id


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

    params = _submitted_task(workspace).python_wheel_task.parameters
    for flag in ("--dataset", "--config", "--summary-out", "--live-status-out"):
        value = params[params.index(flag) + 1]
        assert value.startswith("/Volumes/"), f"{flag} -> {value}"


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


def test_dcs_never_routes_through_run_now_or_a_job_name(settings, dataset):
    """No Serverless routing is introduced: DCS is the same jobs.submit path
    every other compute selection already uses."""
    dcs_settings = settings.model_copy(
        update={"databricks_docker_image_url": "avinashforecastiqacr.azurecr.io/forecastiq-runtime:v1"}
    )
    workspace = _FakeWorkspace()
    assert not hasattr(workspace.jobs, "run_now")

    DatabricksRunner(dcs_settings, workspace_client=workspace).submit(_request(dataset, NEW_COMPUTE))

    assert len(workspace.jobs.submit_calls) == 1


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


def test_execution_does_not_depend_on_a_named_job(settings, dataset):
    """Where a run executes comes from the selection, never a job lookup."""
    workspace, _, _ = _run(settings, dataset, EXISTING)

    assert _submitted_task(workspace).existing_cluster_id == "0826-abc-chosen"
    assert not hasattr(settings, "databricks_job_name")
    assert not hasattr(workspace.jobs, "run_now_calls")


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
        assert task.python_wheel_task.parameters[-2:] == ["--stage", task.task_key]


def test_ray_stages_share_one_job_cluster_not_one_per_task(settings, dataset):
    """Ray parallelism lives inside train/evaluate/explain/rank_select's own
    tasks; the cluster those tasks run on must still be booted once for the
    whole run, not once per task. jobs.submit() (a one-time run) has no
    job_clusters/job_cluster_key support in the installed SDK, so this is
    done by creating one real cluster up front and pointing every task at
    it by existing_cluster_id."""
    workspace, _, _ = _run(settings, dataset, NEW_COMPUTE)

    assert len(workspace.clusters.create_calls) == 1
    created_id = workspace.clusters.created_ids[0]
    for task in _submitted_tasks(workspace):
        assert task.existing_cluster_id == created_id
        assert task.new_cluster is None


def test_the_managed_cluster_self_terminates_and_is_torn_down_on_completion(settings, dataset):
    """Nothing but this run's own tasks ever targets the cluster it creates
    for itself — unlike a real job cluster, Databricks will not tear it
    down on its own, so this runner must (autotermination is only the
    backstop if that is ever missed)."""
    workspace, runner, run_id = _run(settings, dataset, NEW_COMPUTE)

    assert workspace.clusters.create_calls[0]["autotermination_minutes"] == (
        DatabricksRunner._MANAGED_CLUSTER_AUTOTERMINATE_MINUTES
    )

    workspace.jobs.get_run = lambda run_id, **kwargs: SimpleNamespace(
        state=SimpleNamespace(life_cycle_state="TERMINATED", result_state="SUCCESS", state_message=""),
        run_duration=1,
    )
    runner.get_status(run_id)

    assert workspace.clusters.deleted_ids == [workspace.clusters.created_ids[0]]


def test_existing_compute_never_creates_or_deletes_a_cluster(settings, dataset):
    workspace, runner, run_id = _run(settings, dataset, EXISTING)

    workspace.jobs.get_run = lambda run_id, **kwargs: SimpleNamespace(
        state=SimpleNamespace(life_cycle_state="TERMINATED", result_state="SUCCESS", state_message=""),
        run_duration=1,
    )
    runner.get_status(run_id)

    assert workspace.clusters.create_calls == []
    assert workspace.clusters.deleted_ids == []


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
    assert workspace.jobs.submit_calls == []


def test_unknown_compute_mode_is_rejected():
    with pytest.raises(ValueError):
        ComputeSelection(mode="serverless")


# Routing a run by job name is how execution previously reached the
# Serverless pipeline; no application module may do it again.
FORBIDDEN_ROUTING = ("run_now", "_resolve_job_id", "databricks_job_name", "databricks_job_id")


def test_no_application_module_can_route_a_run_by_job_name():
    offenders = []
    for path in Path("backend/app").rglob("*.py"):
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
    settings_source = Path("backend/app/orchestration/databricks_runner.py").read_text()
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

    args = _submitted_task(workspace).python_wheel_task.parameters
    assert f"{artifacts_root}/summary.json" in args
    assert f"{artifacts_root}/live_status.json" in args
