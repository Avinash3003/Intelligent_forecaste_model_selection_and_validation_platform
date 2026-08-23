"""Model-storage results reaching the API envelope.

The engine wrote `model_storage_results` into its run summary, but nothing
mapped it into `PipelineExecutionResult`, so a caller could not tell how
many winning models a run had actually persisted or where they went.
These cover that mapping — including the cases where it must not overstate
what was saved.
"""

from app.orchestration.result_mapper import map_summary_to_result
from app.orchestration.schemas import ExecutionBackend, JobStatus

VOLUME = "/Volumes/forecastiq/forecasting/models_files/runs/run-1"


def _entry(group, model="prophet", persisted=True, error=None):
    return {
        "forecast_group": group,
        "model_name": model,
        "persisted": persisted,
        "uri": f"{VOLUME}/{group.replace(' | ', '_')}_model.pkl" if persisted else None,
        "error": error,
    }


def _map(results):
    return map_summary_to_result(
        run_id="run-1",
        execution_backend=ExecutionBackend.DATABRICKS,
        job_status=JobStatus.COMPLETED,
        summary={"model_storage_results": results} if results is not None else {},
        started_at=None,
        completed_at=None,
        duration_seconds=None,
    )


def _storage(results):
    return _map(results).run_metadata["model_storage"]


def test_the_number_of_saved_models_is_reported():
    storage = _storage([_entry("1 | 1"), _entry("1 | 2"), _entry("1 | 3")])

    assert storage["models_saved"] == 3
    assert storage["groups_total"] == 3


def test_the_storage_location_is_reported():
    assert _storage([_entry("1 | 1")])["location"] == VOLUME


def test_per_key_status_is_reported():
    storage = _storage([_entry("1 | 1", "prophet"), _entry("1 | 2", "seasonal_naive")])

    assert storage["by_group"]["1 | 1"]["model_name"] == "prophet"
    assert storage["by_group"]["1 | 1"]["persisted"] is True
    assert storage["by_group"]["1 | 1"]["uri"].endswith("1_1_model.pkl")
    assert storage["by_group"]["1 | 2"]["model_name"] == "seasonal_naive"


def test_an_unpersisted_key_is_reported_and_not_counted_as_saved():
    """The count must describe files, not attempts."""
    storage = _storage(
        [_entry("1 | 1"), _entry("1 | 2", persisted=False, error="no fitted estimator")]
    )

    assert storage["models_saved"] == 1
    assert storage["groups_total"] == 2
    assert storage["by_group"]["1 | 2"]["persisted"] is False
    assert storage["by_group"]["1 | 2"]["error"] == "no fitted estimator"


def test_a_location_is_only_reported_when_something_was_written():
    storage = _storage([_entry("1 | 1", persisted=False, error="failed")])

    assert storage["models_saved"] == 0
    assert storage["location"] is None


def test_a_run_without_model_storage_reports_an_empty_summary_not_an_error():
    """Local runs and older summaries have no such key."""
    storage = _map(None).run_metadata["model_storage"]

    assert storage == {"models_saved": 0, "groups_total": 0, "location": None, "by_group": {}}


def test_existing_run_metadata_is_unchanged():
    result = map_summary_to_result(
        run_id="run-1",
        execution_backend=ExecutionBackend.DATABRICKS,
        job_status=JobStatus.COMPLETED,
        summary={"run_id": "run-1", "curated_dataset_uri": "/Volumes/x/curated.csv", "group_count": 5},
        started_at=None,
        completed_at=None,
        duration_seconds=None,
    )

    assert result.run_metadata["curated_dataset_uri"] == "/Volumes/x/curated.csv"
    assert result.run_metadata["group_count"] == 5
