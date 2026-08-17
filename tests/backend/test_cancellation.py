"""Run cancellation (Feature 1) and creator attribution (Feature 2).

Covers, per runner:
  * `_cleanup_run_storage()` deletes exactly the cancelled run's own
    storage, never a sibling run's, and is idempotent (safe to call again
    on an already-clean run).
  * `cancel()` is idempotent at the state-machine level: a second call on
    an already-terminal run is a no-op, not an error.
  * `started_by`/`cancelled_by` round-trip through `RunListing`.
  * `MLflowHistoryStore.mark_cancelled()` terminates a RUNNING MLflow run
    as KILLED with the right tags, is idempotent, and is a safe no-op for
    a run whose MLflow row does not exist yet.

API-level RBAC and spoofing protection live in
`test_cancellation_api.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.orchestration.databricks_runner import DatabricksRunner
from app.orchestration.local_runner import LocalRunner, _JobRecord
from app.orchestration.mlflow_history import (
    DATASET_NAME_TAG,
    RUN_ID_TAG,
    STARTED_BY_DISPLAY_NAME_TAG,
    MLflowHistoryStore,
)
from app.orchestration.schemas import JobStatus, PipelineExecutionRequest

from test_databricks_runner import _FakeWorkspace, _request as _dbx_request  # noqa: E402


# ---------------------------------------------------------------------
# LocalRunner cleanup
# ---------------------------------------------------------------------


@pytest.fixture
def local_settings(tmp_path):
    engine_root = tmp_path / "forecast_engine"
    (engine_root / ".venv" / "bin").mkdir(parents=True)
    (engine_root / ".venv" / "bin" / "python").write_text("#!/bin/sh\n")
    return Settings(
        forecast_engine_root=str(engine_root),
        upload_dir=str(tmp_path / "uploads"),
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )


def _write_run_outputs(engine_working_dir: Path, run_id: str, other_run_id: str) -> None:
    for root, suffix in (("curated", ""), ("models", ""), ("artifacts", "")):
        (engine_working_dir / root / run_id).mkdir(parents=True)
        (engine_working_dir / root / run_id / "file.bin").write_text("x")
        (engine_working_dir / root / other_run_id).mkdir(parents=True)
        (engine_working_dir / root / other_run_id / "file.bin").write_text("x")
    (engine_working_dir / "forecasts").mkdir(parents=True)
    (engine_working_dir / "forecasts" / f"{run_id}_forecast.csv").write_text("a,b\n")
    (engine_working_dir / "forecasts" / f"{other_run_id}_forecast.csv").write_text("a,b\n")


def test_local_cleanup_deletes_only_the_cancelled_runs_storage(local_settings, tmp_path):
    runner = LocalRunner(local_settings)
    run_id, other_run_id = "fe-run-aaa111", "fe-run-bbb222"
    engine_working_dir = Path(local_settings.forecast_engine_root).resolve().parent
    _write_run_outputs(engine_working_dir, run_id, other_run_id)

    work_dir = tmp_path / "workdir"
    work_dir.mkdir()
    (work_dir / "summary.json").write_text("{}")

    errors = runner._cleanup_run_storage(run_id, work_dir)

    assert errors == []
    assert not work_dir.exists()
    assert not (engine_working_dir / "curated" / run_id).exists()
    assert not (engine_working_dir / "models" / run_id).exists()
    assert not (engine_working_dir / "artifacts" / run_id).exists()
    assert not (engine_working_dir / "forecasts" / f"{run_id}_forecast.csv").exists()

    # The sibling run is completely untouched.
    assert (engine_working_dir / "curated" / other_run_id / "file.bin").exists()
    assert (engine_working_dir / "models" / other_run_id / "file.bin").exists()
    assert (engine_working_dir / "artifacts" / other_run_id / "file.bin").exists()
    assert (engine_working_dir / "forecasts" / f"{other_run_id}_forecast.csv").exists()


def test_local_cleanup_is_idempotent(local_settings, tmp_path):
    runner = LocalRunner(local_settings)
    run_id = "fe-run-idempotent"
    engine_working_dir = Path(local_settings.forecast_engine_root).resolve().parent
    _write_run_outputs(engine_working_dir, run_id, "fe-run-other")

    first = runner._cleanup_run_storage(run_id, None)
    second = runner._cleanup_run_storage(run_id, None)

    assert first == []
    assert second == []  # nothing left to delete, and that is not an error


def test_local_cancel_on_a_pending_job_is_accepted_and_records_who(local_settings):
    runner = LocalRunner(local_settings)
    record = _JobRecord(run_id="fe-run-pending", status=JobStatus.PENDING, started_at="2026-01-01T00:00:00")
    with runner._lock:
        runner._jobs[record.run_id] = record

    outcome = runner.cancel(record.run_id, cancelled_by_user_id="u-1", cancelled_by_display_name="Jane Doe")

    assert outcome.cancelled is True
    assert record.status is JobStatus.CANCELLED
    assert record.cancelled_by_user_id == "u-1"
    assert record.cancelled_by_display_name == "Jane Doe"


def test_local_cancel_twice_is_a_no_op_the_second_time(local_settings):
    runner = LocalRunner(local_settings)
    record = _JobRecord(run_id="fe-run-twice", status=JobStatus.RUNNING, started_at="2026-01-01T00:00:00")
    with runner._lock:
        runner._jobs[record.run_id] = record

    first = runner.cancel(record.run_id, "u-1", "Jane Doe")
    second = runner.cancel(record.run_id, "u-2", "Someone Else")

    assert first.cancelled is True
    assert second.cancelled is False
    # The second (no-op) call must not overwrite who actually cancelled it.
    assert record.cancelled_by_display_name == "Jane Doe"


def test_local_started_by_and_cancelled_by_round_trip_through_run_listing(local_settings):
    runner = LocalRunner(local_settings)
    record = _JobRecord(
        run_id="fe-run-attrib",
        status=JobStatus.RUNNING,
        started_at="2026-01-01T00:00:00",
        started_by_user_id="u-1",
        started_by_display_name="Avinash Reddy",
    )
    with runner._lock:
        runner._jobs[record.run_id] = record

    runner.cancel(record.run_id, "u-2", "Admin User")
    listing = runner.get_run(record.run_id)

    assert listing.started_by == "Avinash Reddy"
    assert listing.cancelled_by == "Admin User"
    assert listing.job_status is JobStatus.CANCELLED


# ---------------------------------------------------------------------
# DatabricksRunner cleanup
# ---------------------------------------------------------------------


@pytest.fixture
def dbx_settings(tmp_path):
    return Settings(
        execution_mode="databricks",
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


def test_databricks_cleanup_deletes_only_the_cancelled_runs_storage(dbx_settings, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(dbx_settings, workspace_client=workspace)
    run_id = runner.submit(_dbx_request(dataset))
    other_run_id = "dbx-run-sibling"

    # Simulate outputs a further-along run would have written, for both
    # this run and an unrelated sibling.
    for run in (run_id, other_run_id):
        workspace.files.uploaded[f"/Volumes/forecastiq/forecasting/curated_files/runs/{run}/curated.parquet"] = b"x"
        workspace.files.uploaded[f"/Volumes/forecastiq/forecasting/models_files/runs/{run}/1_1_model.pkl"] = b"x"
        workspace.files.uploaded[f"/Volumes/forecastiq/forecasting/artifacts_files/runs/{run}/insights.json"] = b"x"
        workspace.files.uploaded[f"/Volumes/forecastiq/forecasting/forecasts_files/runs/{run}_forecast.csv"] = b"x"

    outcome = runner.cancel(run_id, "u-1", "Jane Doe")

    assert outcome.cancelled is True
    assert outcome.cleanup_errors == []

    remaining = workspace.files.uploaded
    assert not any(run_id in key and "sibling" not in key for key in remaining if other_run_id not in key and run_id in key)
    # This run's paths are gone...
    for key in list(remaining):
        assert f"/{run_id}/" not in key and not key.startswith(f"/Volumes/forecastiq/forecasting/forecasts_files/runs/{run_id}_")
    # ...the sibling's are untouched.
    assert f"/Volumes/forecastiq/forecasting/curated_files/runs/{other_run_id}/curated.parquet" in remaining
    assert f"/Volumes/forecastiq/forecasting/models_files/runs/{other_run_id}/1_1_model.pkl" in remaining
    assert f"/Volumes/forecastiq/forecasting/artifacts_files/runs/{other_run_id}/insights.json" in remaining
    assert f"/Volumes/forecastiq/forecasting/forecasts_files/runs/{other_run_id}_forecast.csv" in remaining


def test_databricks_cleanup_is_idempotent(dbx_settings, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(dbx_settings, workspace_client=workspace)
    run_id = runner.submit(_dbx_request(dataset))

    first_errors = runner._cleanup_run_storage(run_id)
    second_errors = runner._cleanup_run_storage(run_id)

    assert first_errors == []
    assert second_errors == []


def test_databricks_cancel_refuses_a_run_that_already_completed_on_databricks(dbx_settings, dataset):
    """cancel() must refresh from the workspace before checking terminality.

    A run's in-memory record is only updated when something reads it
    (DatabricksRunner._refresh's own contract) — if the job finished on
    Databricks since the last poll, the record can still read RUNNING.
    Without a refresh first, cancel() would send a (harmless) cancel_run
    call and then unconditionally delete this completed run's curated
    dataset, models and forecast export via _cleanup_run_storage, and mark
    a real success as CANCELLED in MLflow.
    """
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(dbx_settings, workspace_client=workspace)
    run_id = runner.submit(_dbx_request(dataset))

    # The job finished successfully on Databricks, but nothing has polled
    # get_status()/get_result() since — the in-memory record is still
    # whatever submit() left it as.
    workspace.jobs._state = SimpleNamespace(life_cycle_state="TERMINATED", result_state="SUCCESS", state_message="")
    workspace.files.uploaded[
        f"/Volumes/forecastiq/forecasting/curated_files/runs/{run_id}/curated.parquet"
    ] = b"real completed output"

    outcome = runner.cancel(run_id)

    assert outcome.cancelled is False
    assert workspace.jobs.cancelled == []  # never asked Databricks to cancel a finished run
    # The completed run's real output survives.
    assert f"/Volumes/forecastiq/forecasting/curated_files/runs/{run_id}/curated.parquet" in workspace.files.uploaded


def test_databricks_started_by_travels_through_the_job_configuration(dbx_settings, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(dbx_settings, workspace_client=workspace)
    request = PipelineExecutionRequest(
        dataset_path=str(dataset),
        dataset_name="sales.csv",
        forecast_configuration={
            "date_column": "date", "target_column": "sales",
            "key_columns": ["store"], "feature_columns": [], "aggregation_method": "sum",
        },
        started_by_user_id="u-1",
        started_by_display_name="Avinash Reddy",
    )
    run_id = runner.submit(request)

    config_bytes = workspace.files.uploaded[f"/Volumes/forecastiq/forecasting/forecast_files/runs/{run_id}/forecast_configuration.json"]
    payload = json.loads(config_bytes)
    assert payload["started_by_user_id"] == "u-1"
    assert payload["started_by_display_name"] == "Avinash Reddy"


# ---------------------------------------------------------------------
# MLflowHistoryStore.mark_cancelled
# ---------------------------------------------------------------------


@pytest.fixture
def mlflow_settings(tmp_path):
    return Settings(
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        mlflow_experiment_name="/forecast-engine-test",
        upload_dir=str(tmp_path / "uploads"),
    )


def _open_real_mlflow_run(settings: Settings, run_id: str, started_by: str) -> None:
    """Opens a Parent Run the same way `begin()` does, and — deliberately —
    never closes it, exactly what a forced kill leaves behind. Uses
    `MlflowClient` directly rather than the fluent `mlflow.start_run()` API:
    the fluent API tracks one process-global "active run" stack, which a
    run meant to stay open forever (never `end_run()`-ed) leaves dangling
    for every later fluent call in the same test process.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri_resolved)
    experiment_name = settings.mlflow_experiment_name_resolved
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)

    run = client.create_run(experiment_id, run_name=f"forecast-run-{run_id}")
    client.set_tag(run.info.run_id, RUN_ID_TAG, run_id)
    client.set_tag(run.info.run_id, DATASET_NAME_TAG, "sales.csv")
    client.set_tag(run.info.run_id, STARTED_BY_DISPLAY_NAME_TAG, started_by)


def test_mark_cancelled_terminates_a_running_mlflow_run_as_killed(mlflow_settings):
    store = MLflowHistoryStore(mlflow_settings)
    _open_real_mlflow_run(mlflow_settings, "fe-run-mlflow-1", "Avinash Reddy")

    found = store.mark_cancelled("fe-run-mlflow-1", "u-2", "Admin User", "2026-08-12T10:00:00")
    assert found is True

    listing = store.get_listing("fe-run-mlflow-1")
    assert listing is not None
    assert listing.job_status is JobStatus.CANCELLED
    assert listing.started_by == "Avinash Reddy"
    assert listing.cancelled_by == "Admin User"


def test_mark_cancelled_is_idempotent(mlflow_settings):
    store = MLflowHistoryStore(mlflow_settings)
    _open_real_mlflow_run(mlflow_settings, "fe-run-mlflow-2", "Someone")

    first = store.mark_cancelled("fe-run-mlflow-2", "u-1", "Jane", "2026-08-12T10:00:00")
    second = store.mark_cancelled("fe-run-mlflow-2", "u-1", "Jane", "2026-08-12T10:00:05")

    assert first is True
    assert second is True  # re-affirming an already-KILLED run is not an error


def test_mark_cancelled_on_a_run_that_never_reached_mlflow_is_a_safe_no_op(mlflow_settings):
    store = MLflowHistoryStore(mlflow_settings)
    found = store.mark_cancelled("fe-run-never-existed", "u-1", "Jane", "2026-08-12T10:00:00")
    assert found is False
