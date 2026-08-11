"""forecast_export_result / artifacts_mirror_result reaching the API envelope.

Same shape as test_model_storage_metadata.py: the engine already writes
both fields into its run summary; these cover that result_mapper.py
surfaces them into run_metadata without inventing or dropping anything.
"""

from app.orchestration.result_mapper import map_summary_to_result
from app.orchestration.schemas import ExecutionBackend, JobStatus


def _map(summary):
    return map_summary_to_result(
        run_id="run-1",
        execution_backend=ExecutionBackend.DATABRICKS,
        job_status=JobStatus.COMPLETED,
        summary=summary,
        started_at=None,
        completed_at=None,
        duration_seconds=None,
    )


def test_forecast_export_result_is_surfaced_verbatim():
    export = {"enabled": True, "persisted": True, "uri": "/Volumes/.../run-1_forecast.csv", "rows": 42, "error": None}

    result = _map({"forecast_export_result": export})

    assert result.run_metadata["forecast_export"] == export


def test_artifacts_mirror_result_is_surfaced_verbatim():
    mirror = {"enabled": True, "persisted": [{"file": "business_insights.json", "persisted": True, "uri": "x", "error": None}]}

    result = _map({"artifacts_mirror_result": mirror})

    assert result.run_metadata["artifacts_mirror"] == mirror


def test_missing_fields_default_to_empty_not_an_error():
    result = _map({})

    assert result.run_metadata["forecast_export"] == {}
    assert result.run_metadata["artifacts_mirror"] == {}


def test_existing_run_metadata_fields_are_unaffected():
    result = _map({"curated_dataset_uri": "/Volumes/x/curated.csv", "group_count": 5})

    assert result.run_metadata["curated_dataset_uri"] == "/Volumes/x/curated.csv"
    assert result.run_metadata["group_count"] == 5
