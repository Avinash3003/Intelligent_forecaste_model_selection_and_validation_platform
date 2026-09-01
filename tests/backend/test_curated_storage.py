"""Curated dataset storage: where cloud runs write it, and who can read it.

The defect these cover: the engine's curated root is a *relative* path, so a
Databricks driver resolved it against a working directory the job destroys
on exit. The curated dataset was written successfully and then lost, and the
Results page — which reads that path as a local file on the API host, where
it never existed — reported "curated storage may be disabled" instead.
"""

import json
import dataclasses
from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.orchestration.databricks_runner import DatabricksRunner
from app.schemas.compute import ComputeSelection
from app.orchestration.schemas import PipelineExecutionRequest
from app.services.dataset_preview_service import DatasetPreviewService


class _FakeFiles:
    def __init__(self) -> None:
        self.uploaded: dict[str, bytes] = {}

    def upload(self, file_path, contents, overwrite=None):
        self.uploaded[file_path] = contents.read()

    def download(self, file_path):
        if file_path not in self.uploaded:
            raise FileNotFoundError(file_path)
        return SimpleNamespace(contents=SimpleNamespace(read=lambda: self.uploaded[file_path]))



# The per-run paths this run was started with. The task definition holds
# `{{job.parameters.x}}` placeholders; run_now supplies the real values.
def _submitted_parameters(workspace):
    return workspace.jobs.run_now_calls[-1]["job_parameters"]


class _FakeJobs:
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
        return SimpleNamespace(run_id=7)

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
        execution_mode="databricks",
        databricks_host="https://example.invalid",
        databricks_token="t",
        databricks_uploads_volumes_root="/Volumes/cat/sch/upload_files",
        databricks_curated_volumes_root="/Volumes/cat/sch/curated_files",
        databricks_models_volumes_root="/Volumes/cat/sch/models_files",
        databricks_artifacts_volumes_root="/Volumes/cat/sch/artifacts_files",
        upload_dir=str(tmp_path / "uploads"),
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("date,store,sales\n2024-01-01,1,10\n")
    return path


def _request(dataset, run_id="dbx-run-curated"):
    return PipelineExecutionRequest(
        compute=ComputeSelection(mode="existing_compute", cluster_id="test-cluster"),
        run_id=run_id,
        dataset_path=str(dataset),
        dataset_name="sales.csv",
        forecast_configuration={"date_column": "date", "target_column": "sales", "key_columns": ["store"]},
    )


def _submitted_config(workspace, run_id):
    root = f"/Volumes/cat/sch/artifacts_files/runs/{run_id}"
    return json.loads(workspace.files.uploaded[f"{root}/forecast_configuration.json"])


# ---------------------------------------------------------------------
# The curated path the engine is told to use
# ---------------------------------------------------------------------


def test_cloud_run_is_given_an_absolute_curated_path(settings, dataset):
    """The defect: without this the engine used its relative default."""
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset))

    root = _submitted_config(workspace, run_id)["curated_storage"]["root_dir"]
    assert root.startswith("/Volumes/"), "a relative root is what made the dataset ephemeral"
    # The engine appends the run id under this root, so it must not appear twice.
    assert root == "/Volumes/cat/sch/curated_files/runs"
    assert root.count(run_id) == 0


def test_curated_output_is_partitioned_by_run(settings, dataset):
    # One run must never overwrite another's curated dataset.
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    a = runner.submit(_request(dataset, run_id="dbx-run-aaa"))
    b = runner.submit(_request(dataset, run_id="dbx-run-bbb"))

    # The engine writes <root>/<run_id>/..., so a shared root still keeps
    # each run's curated dataset in its own directory.
    assert a != b
    for run_id in (a, b):
        root = _submitted_config(workspace, run_id)["curated_storage"]["root_dir"]
        assert root == "/Volumes/cat/sch/curated_files/runs"


def test_curated_volume_is_separate_from_the_uploads_volume(settings, dataset):
    # Raw uploads and derived datasets keep their own containers.
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset))

    config = _submitted_config(workspace, run_id)
    assert config["curated_storage"]["root_dir"].startswith("/Volumes/cat/sch/curated_files/")
    assert _submitted_parameters(workspace)["dataset"].startswith("/Volumes/cat/sch/upload_files/")



def test_existing_run_file_layout_is_unchanged(settings, dataset):
    # The curated addition must not disturb the four job parameters.
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset))

    upload_root = f"/Volumes/cat/sch/upload_files/runs/{run_id}"
    artifacts_root = f"/Volumes/cat/sch/artifacts_files/runs/{run_id}"
    parameters = _submitted_parameters(workspace)
    assert parameters["dataset"] == f"{upload_root}/sales.csv"
    assert parameters["config"] == f"{artifacts_root}/forecast_configuration.json"
    assert parameters["summary_out"] == f"{artifacts_root}/summary.json"
    assert parameters["live_status_out"] == f"{artifacts_root}/live_status.json"


# ---------------------------------------------------------------------
# Reading it back for the Results page
# ---------------------------------------------------------------------


class _Executor:
    def __init__(self, uri, runner=None):
        self._uri = uri
        self._runner = runner

    def get_result(self, run_id):
        return SimpleNamespace(run_metadata={"curated_dataset_uri": self._uri})


def test_preview_reads_a_local_curated_file(tmp_path):
    # Local execution is unchanged: the path is a real file on this host.
    curated = tmp_path / "curated.csv"
    curated.write_text("month,sales\n2024-01,10\n2024-02,12\n")
    service = DatasetPreviewService(executor=_Executor(str(curated)))

    preview = service.get_preview("run-local")
    assert preview.available is True
    assert preview.columns == ["month", "sales"]
    assert preview.total_rows == 2


def test_preview_reads_a_uc_volume_file_through_the_workspace(settings):
    """The defect: this path is not a local file, so the preview reported
    the dataset missing even though the run had written it."""
    workspace = _FakeWorkspace()
    uri = "/Volumes/cat/sch/curated_files/runs/dbx-run-x/curated.csv"
    workspace.files.uploaded[uri] = b"month,sales\n2024-01,10\n"
    runner = DatabricksRunner(settings, workspace_client=workspace)

    service = DatasetPreviewService(executor=_Executor(uri, runner=runner))
    preview = service.get_preview("dbx-run-x")

    assert preview.available is True
    assert preview.columns == ["month", "sales"]
    assert preview.total_rows == 1


def test_preview_is_unavailable_rather_than_raising_when_unreadable(settings):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    service = DatasetPreviewService(
        executor=_Executor("/Volumes/cat/sch/curated_files/runs/gone/curated.csv", runner=runner)
    )
    assert service.get_preview("gone").available is False


def test_preview_without_a_databricks_runner_degrades_cleanly(tmp_path):
    # A local runner has no workspace client to fall back to.
    service = DatasetPreviewService(executor=_Executor("/Volumes/cat/sch/curated_files/x.csv"))
    assert service.get_preview("run").available is False


# ---------------------------------------------------------------------
# Winning-model storage: the same contract, its own volume
# ---------------------------------------------------------------------


def test_cloud_run_is_given_an_absolute_models_path(settings, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset))

    root = _submitted_config(workspace, run_id)["model_storage"]["root_dir"]
    assert root == "/Volumes/cat/sch/models_files/runs"
    # The engine appends the run id, which is what separates runs.
    assert run_id not in root


def test_models_use_their_own_volume_not_the_curated_one(settings, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset))

    config = _submitted_config(workspace, run_id)
    assert config["model_storage"]["root_dir"].startswith("/Volumes/cat/sch/models_files/")
    assert config["curated_storage"]["root_dir"].startswith("/Volumes/cat/sch/curated_files/")


