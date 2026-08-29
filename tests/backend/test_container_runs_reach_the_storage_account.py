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
    """The Workspace Files API, which is a different API on the same client
    from the Volumes one — a container run's staging goes through it."""

    def __init__(self, uploaded):
        self._uploaded = uploaded
        self.made = []

    def mkdirs(self, path):
        self.made.append(path)

    def upload(self, path, content, format=None, overwrite=False):
        self._uploaded[path] = content


class _Workspace(_FakeWorkspace):
    def __init__(self, state=None):
        super().__init__(state)
        # One dict for both APIs: a test asserts *where* a file landed, and
        # the path itself is what says which API owned it.
        self.workspace = _FakeWorkspaceApi(self.files.uploaded)

VOLUME = "/Volumes/forecastiq/forecasting"
STAGING = "/Workspace/Shared/forecastiq/runs"


def _settings(tmp_path, mlflow_db, *, image: str | None):
    return Settings(
        execution_mode="databricks",
        databricks_host="https://example.invalid",
        databricks_token="test-token",
        databricks_volumes_root=f"{VOLUME}/forecast_files",
        databricks_curated_volumes_root=f"{VOLUME}/curated_files",
        databricks_models_volumes_root=f"{VOLUME}/models_files",
        databricks_forecasts_volumes_root=f"{VOLUME}/forecasts_files",
        databricks_artifacts_volumes_root=f"{VOLUME}/artifacts_files",
        databricks_workspace_staging_root=STAGING,
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
    job_compute=JobComputeConfig(node_type_id="Standard_DS3_v2", runtime_key="15.4"),
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

    assert config_key.startswith(f"{VOLUME}/forecast_files/runs/")
    assert payload["curated_storage"]["root_dir"] == f"{VOLUME}/curated_files/runs"
    assert payload["model_storage"]["root_dir"] == f"{VOLUME}/models_files/runs"
    assert payload["forecast_export"]["root_dir"] == f"{VOLUME}/forecasts_files/runs"
    assert payload["artifacts_mirror"]["root_dir"] == f"{VOLUME}/artifacts_files/runs"
    # It writes straight to its final home, so it has nothing to copy.
    assert "volume_sync" not in payload


def test_a_job_compute_run_without_an_image_also_uses_volumes(tmp_path, mlflow_db, dataset):
    _, _, config_key, payload = _config_for(tmp_path, mlflow_db, dataset, JOB, None)

    assert config_key.startswith(f"{VOLUME}/forecast_files/runs/")
    assert "volume_sync" not in payload


# --- the container run: staged where it can write ----------------------


def test_a_container_run_stages_to_the_workspace_because_it_has_no_mount(tmp_path, mlflow_db, dataset):
    _, run_id, config_key, payload = _config_for(tmp_path, mlflow_db, dataset, JOB, "acr.io/forecastiq:1")

    assert config_key == f"{STAGING}/{run_id}/forecast_configuration.json"
    assert payload["curated_storage"]["root_dir"] == f"{STAGING}/curated"
    assert payload["model_storage"]["root_dir"] == f"{STAGING}/models"


# --- defect 2: and copies itself back into the storage account ---------


def test_a_container_run_is_told_to_copy_every_output_into_the_volume(tmp_path, mlflow_db, dataset):
    _, run_id, _, payload = _config_for(tmp_path, mlflow_db, dataset, JOB, "acr.io/forecastiq:1")

    pairs = {t["source"]: t["destination"] for t in payload["volume_sync"]["targets"]}

    assert pairs == {
        f"{STAGING}/{run_id}": f"{VOLUME}/forecast_files/runs/{run_id}",
        f"{STAGING}/curated/{run_id}": f"{VOLUME}/curated_files/runs/{run_id}",
        f"{STAGING}/models/{run_id}": f"{VOLUME}/models_files/runs/{run_id}",
        f"{STAGING}/artifacts/{run_id}": f"{VOLUME}/artifacts_files/runs/{run_id}",
        f"{STAGING}/forecasts/{run_id}_forecast.csv": f"{VOLUME}/forecasts_files/runs/{run_id}_forecast.csv",
    }


def test_every_output_the_engine_is_given_is_also_a_sync_source(tmp_path, mlflow_db, dataset):
    """The guard against a future output being added and quietly never
    reaching the storage account — the exact shape of this whole bug."""
    _, run_id, config_key, payload = _config_for(tmp_path, mlflow_db, dataset, JOB, "acr.io/forecastiq:1")

    sources = {t["source"] for t in payload["volume_sync"]["targets"]}
    written_roots = {
        payload["curated_storage"]["root_dir"],
        payload["model_storage"]["root_dir"],
        payload["forecast_export"]["root_dir"],
        payload["artifacts_mirror"]["root_dir"],
        config_key.rsplit("/", 1)[0],
    }

    for root in written_roots:
        assert any(source.startswith(root) for source in sources), f"{root} is written but never synced"


def test_the_sync_destination_is_exactly_where_a_normal_run_would_have_written(
    tmp_path, mlflow_db, dataset
):
    """A container run and an existing-compute run must leave the storage
    account in the same state; only the route differs."""
    _, run_id, direct_key, direct = _config_for(tmp_path, mlflow_db, dataset, EXISTING, "acr.io/forecastiq:1")
    _, _, _, container = _config_for(tmp_path, mlflow_db, dataset, JOB, "acr.io/forecastiq:1")

    destinations = {t["destination"] for t in container["volume_sync"]["targets"]}

    assert f"{direct['curated_storage']['root_dir']}/{run_id}" in destinations
    assert f"{direct['model_storage']['root_dir']}/{run_id}" in destinations
    assert f"{direct['artifacts_mirror']['root_dir']}/{run_id}" in destinations
    assert f"{direct['forecast_export']['root_dir']}/{run_id}_forecast.csv" in destinations
    assert direct_key.rsplit("/", 1)[0] in destinations
