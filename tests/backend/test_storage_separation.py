"""Uploads hold ONLY the original dataset. Everything else the run
produces — config, summary, live status, the registry breadcrumb — lives
under the artifacts volume instead, in the same per-run folder MLflow's
own artifact mirror already uses.

Before this, `_run_root` served both roles: the uploads volume held the
raw dataset AND the run's config/summary/status, so "uploads" was never
just user input.
"""

from __future__ import annotations

import pytest

from app.config.settings import Settings
from app.orchestration.databricks_runner import DatabricksRunner
from test_databricks_runner import _FakeWorkspace, _request


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("date,store,sales\n2024-01-01,1,10\n")
    return path


def _settings(tmp_path, **overrides):
    fields = {
        "execution_mode": "databricks",
        "databricks_host": "https://example.invalid",
        "databricks_token": "test-token",
        "databricks_uploads_volumes_root": "/Volumes/forecastiq/forecasting/upload_files",
        "databricks_artifacts_volumes_root": "/Volumes/forecastiq/forecasting/artifacts_files",
        "mlflow_tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
    }
    fields.update(overrides)
    return Settings(**fields)


def test_uploads_volume_holds_only_the_original_dataset(tmp_path, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(_settings(tmp_path), workspace_client=workspace)

    run_id = runner.submit(_request(dataset))

    upload_prefix = f"/Volumes/forecastiq/forecasting/upload_files/runs/{run_id}/"
    uploaded_under_uploads = [p for p in workspace.files.uploaded if p.startswith(upload_prefix)]
    assert uploaded_under_uploads == [f"{upload_prefix}sales.csv"]


def test_run_artifacts_live_under_the_artifacts_volume_not_uploads(tmp_path, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(_settings(tmp_path), workspace_client=workspace)

    run_id = runner.submit(_request(dataset))

    artifacts_prefix = f"/Volumes/forecastiq/forecasting/artifacts_files/runs/{run_id}/"
    artifact_files = {p[len(artifacts_prefix):] for p in workspace.files.uploaded if p.startswith(artifacts_prefix)}
    assert artifact_files == {"forecast_configuration.json", "registry.json"}


def test_job_parameters_point_summary_and_status_at_artifacts(tmp_path, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(_settings(tmp_path), workspace_client=workspace)

    run_id = runner.submit(_request(dataset))

    task = workspace.jobs.submit_calls[0]["tasks"][0]
    args = task.python_wheel_task.parameters
    artifacts_root = f"/Volumes/forecastiq/forecasting/artifacts_files/runs/{run_id}"
    assert f"{artifacts_root}/summary.json" in args
    assert f"{artifacts_root}/live_status.json" in args
    assert f"{artifacts_root}/forecast_configuration.json" in args


def test_cleanup_removes_the_upload_and_the_run_artifacts_folder(tmp_path, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(_settings(tmp_path), workspace_client=workspace)
    run_id = runner.submit(_request(dataset))

    errors = runner._cleanup_run_storage(run_id, uses_container=False)

    assert errors == []
    remaining = [p for p in workspace.files.uploaded if run_id in p]
    assert remaining == []


# --- the deployed App Service configures uploads with the OLD name -------
#
# Deleting these two once already broke production: the deployed app sets
# DATABRICKS_VOLUMES_ROOT, pydantic's extra="ignore" dropped it silently,
# and every upload went to a volume that does not exist.


def test_the_old_settings_field_name_still_configures_uploads(tmp_path, dataset):
    settings = _settings(
        tmp_path,
        databricks_uploads_volumes_root=None,
        databricks_volumes_root="/Volumes/legacy/forecasting/forecast_files",
    )
    assert settings.databricks_uploads_volumes_root == "/Volumes/legacy/forecasting/forecast_files"

    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_request(dataset))

    assert f"/Volumes/legacy/forecasting/forecast_files/runs/{run_id}/sales.csv" in workspace.files.uploaded


def test_a_new_explicit_uploads_root_wins_over_the_old_field(tmp_path):
    settings = _settings(
        tmp_path,
        databricks_uploads_volumes_root="/Volumes/new/forecasting/upload_files",
        databricks_volumes_root="/Volumes/legacy/forecasting/forecast_files",
    )
    assert settings.databricks_uploads_volumes_root == "/Volumes/new/forecasting/upload_files"


def test_the_default_uploads_volume_is_the_one_that_exists_in_databricks(tmp_path):
    """The default has to track the real volume name, not the intended one.
    The rename to upload_files needs MANAGE on the volume, which this app's
    service principal does not have — so until an owner renames it, a
    default of upload_files points at nothing and fails every upload."""
    settings = Settings(_env_file=None, mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")

    assert settings.databricks_uploads_volumes_root.endswith("/forecast_files")
