"""Container runs must still land their outputs in the storage account.

Two defects are pinned here, both from the change that first taught this
runner about Databricks Container Services.

1. A container has no UC Volumes mount, so its runs were routed to
   workspace staging instead. Correct — but the routing asked "is a
   container image *configured*", not "does *this run* use one". The image
   is only ever attached to a job cluster this runner creates, so
   existing-compute runs execute on a normal runtime with a working mount
   and were being diverted for no reason at all.

2. Nothing then brought a real container run's outputs back. Workspace
   files are not the storage account, so run summaries, curated data,
   models, forecast CSVs and artifacts silently stopped arriving in it
   while the runs reported success.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from app.config.settings import Settings  # noqa: E402
from app.orchestration.databricks_runner import DatabricksRunner  # noqa: E402
from app.orchestration.schemas import PipelineExecutionRequest  # noqa: E402
from app.schemas.compute import ComputeSelection, JobComputeConfig  # noqa: E402

from test_databricks_runner import _FakeWorkspace  # noqa: E402


class _FakeWorkspaceApi:
    """The Workspace Files API — a different API on the same client from the
    Volumes one.

    Keeps its OWN record, deliberately. Sharing one dict with the Volumes
    fake made "was this written to the workspace?" unanswerable, and that is
    now the security question the tests exist to answer."""

    def __init__(self):
        self.uploaded: dict[str, object] = {}
        self.made = []

    def mkdirs(self, path):
        self.made.append(path)

    def upload(self, path, content, format=None, overwrite=False):
        self.uploaded[path] = content


class _Workspace(_FakeWorkspace):
    def __init__(self, state=None):
        super().__init__(state)
        # One dict for both APIs: a test asserts *where* a file landed, and
        # the path itself is what says which API owned it.
        self.workspace = _FakeWorkspaceApi()

VOLUME = "/Volumes/forecastiq/forecasting"


def _settings(tmp_path, mlflow_db, *, image: str | None):
    return Settings(
        execution_mode="databricks",
        databricks_host="https://example.invalid",
        databricks_token="test-token",
        databricks_uploads_volumes_root=f"{VOLUME}/upload_files",
        databricks_curated_volumes_root=f"{VOLUME}/curated_files",
        databricks_models_volumes_root=f"{VOLUME}/models_files",
        databricks_forecasts_volumes_root=f"{VOLUME}/forecasts_files",
        databricks_artifacts_volumes_root=f"{VOLUME}/artifacts_files",
        databricks_docker_image_url=image or "",
        upload_dir=str(tmp_path / "uploads"),
        mlflow_tracking_uri=f"sqlite:///{mlflow_db}",
    )


def _request(dataset, compute):
    return PipelineExecutionRequest(
        compute=compute,
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
        horizon=18,
    )


EXISTING = ComputeSelection(mode="existing_compute", cluster_id="test-cluster")
JOB = ComputeSelection(
    mode="new_job_compute",
    job_compute=JobComputeConfig(node_type_id="Standard_DS3_v2", runtime_key="15.4.x-cpu-ml-scala2.12"),
)


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("date,store,sales\n2024-01-01,1,10\n")
    return path


def _config_for(tmp_path, mlflow_db, dataset, compute, image):
    runner = DatabricksRunner(_settings(tmp_path, mlflow_db, image=image), workspace_client=_Workspace())
    run_id = runner.submit(_request(dataset, compute))
    workspace = runner._workspace
    key = next(k for k in workspace.files.uploaded if k.endswith("forecast_configuration.json"))
    return runner, run_id, key, json.loads(workspace.files.uploaded[key])


# --- defect 1: existing compute was diverted for no reason -------------


def test_an_existing_compute_run_still_uses_volumes_when_an_image_is_configured(tmp_path, mlflow_db, dataset):
    """The image is never attached to an existing cluster, so that run has
    a working mount and belongs in the storage account, as it always did."""
    _, _, config_key, payload = _config_for(tmp_path, mlflow_db, dataset, EXISTING, "acr.io/forecastiq:1")

    assert config_key.startswith(f"{VOLUME}/artifacts_files/runs/")
    assert payload["curated_storage"]["root_dir"] == f"{VOLUME}/curated_files/runs"
    assert payload["model_storage"]["root_dir"] == f"{VOLUME}/models_files/runs"
    assert payload["forecast_export"]["root_dir"] == f"{VOLUME}/forecasts_files/runs"
    assert payload["artifacts_mirror"]["root_dir"] == f"{VOLUME}/artifacts_files/runs"
    # It writes straight to its final home, so it has nothing to copy.
    assert "volume_sync" not in payload


def test_a_job_compute_run_without_an_image_also_uses_volumes(tmp_path, mlflow_db, dataset):
    _, _, config_key, payload = _config_for(tmp_path, mlflow_db, dataset, JOB, None)

    assert config_key.startswith(f"{VOLUME}/artifacts_files/runs/")
    assert "volume_sync" not in payload


# --- the container run: staged straight into Unity Catalog -------------


def test_a_container_run_stages_into_the_volume_not_the_workspace(tmp_path, mlflow_db, dataset):
    _, run_id, config_key, payload = _config_for(tmp_path, mlflow_db, dataset, JOB, "acr.io/forecastiq:1")

    assert config_key == f"{VOLUME}/artifacts_files/runs/{run_id}/forecast_configuration.json"
    assert payload["curated_storage"]["root_dir"] == f"{VOLUME}/curated_files/runs"
    assert payload["model_storage"]["root_dir"] == f"{VOLUME}/models_files/runs"


# --- the architecture that replaced the sync step ----------------------


def test_both_execution_modes_receive_identical_unity_catalog_paths(tmp_path, mlflow_db, dataset):
    """The whole point of the storage adapter: the runner stops caring which
    compute is running, because reaching a volume is the adapter's problem.

    Before this there were two path layouts -- UC for existing compute,
    workspace staging for containers -- and a copy step to reconcile them.
    That copy is what duplicated sensitive data under an Analyst-readable
    workspace folder, and what silently delivered nothing when its
    authentication broke."""
    _, run_id, existing_key, existing = _config_for(tmp_path, mlflow_db, dataset, EXISTING, "acr.io/forecastiq:1")
    _, _, container_key, container = _config_for(tmp_path, mlflow_db, dataset, JOB, "acr.io/forecastiq:1")

    assert existing_key == container_key
    for block in ("curated_storage", "model_storage", "forecast_export", "artifacts_mirror"):
        assert existing[block]["root_dir"] == container[block]["root_dir"], block
        assert container[block]["root_dir"].startswith(f"{VOLUME}/"), block


def test_a_container_run_is_never_asked_to_copy_anything(tmp_path, mlflow_db, dataset):
    """No `volume_sync` block: the engine already wrote to the source of
    truth, so there is nothing left to copy. The copy step is inert rather
    than deleted, so one revert restores the old behaviour."""
    _, _, _, payload = _config_for(tmp_path, mlflow_db, dataset, JOB, "acr.io/forecastiq:1")

    assert "volume_sync" not in payload


def test_no_run_data_is_staged_in_the_workspace(tmp_path, mlflow_db, dataset):
    """The security invariant. Workspace ACLs inherit from /Shared and
    cannot be denied, so a copy of a dataset or a model binary there is
    reachable by anyone the UC grants deliberately exclude."""
    runner = DatabricksRunner(
        _settings(tmp_path, mlflow_db, image="acr.io/forecastiq:1"), workspace_client=_Workspace()
    )
    run_id = runner.submit(_request(dataset, JOB))
    workspace = runner._workspace

    assert list(workspace.files.uploaded), "the run must stage to the volume"
    assert all(p.startswith("/Volumes/") for p in workspace.files.uploaded)
    assert not list(workspace.workspace.uploaded), "nothing may be written to the workspace"


def test_every_engine_output_root_is_a_unity_catalog_volume(tmp_path, mlflow_db, dataset):
    """Guards the shape of the bug: one output quietly keeping a separate
    path is exactly how curated data and models ended up outside UC."""
    _, _, config_key, payload = _config_for(tmp_path, mlflow_db, dataset, JOB, "acr.io/forecastiq:1")

    roots = [
        config_key.rsplit("/", 1)[0],
        payload["curated_storage"]["root_dir"],
        payload["model_storage"]["root_dir"],
        payload["forecast_export"]["root_dir"],
        payload["artifacts_mirror"]["root_dir"],
    ]
    for root in roots:
        assert root.startswith("/Volumes/"), root
