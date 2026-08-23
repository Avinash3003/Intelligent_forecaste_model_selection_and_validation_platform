"""Priority #8 — dataset date coverage travels from the run's own
`quality_report` (computed from the raw dataset's date column, never the
upload/run time) through `map_summary_to_result` into `run_metadata`.
"""

from __future__ import annotations

from app.orchestration.result_mapper import map_summary_to_result
from app.orchestration.schemas import ExecutionBackend, JobStatus


def _map(summary: dict) -> dict:
    result = map_summary_to_result(
        run_id="run-1",
        execution_backend=ExecutionBackend.LOCAL,
        job_status=JobStatus.COMPLETED,
        summary=summary,
        started_at=None,
        completed_at=None,
        duration_seconds=None,
    )
    return result.run_metadata


def test_date_range_is_read_from_the_quality_report():
    metadata = _map(
        {"quality_report": {"date_range_start": "2021-01-01T00:00:00", "date_range_end": "2025-02-15T00:00:00"}}
    )
    assert metadata["date_range_start"] == "2021-01-01T00:00:00"
    assert metadata["date_range_end"] == "2025-02-15T00:00:00"


def test_missing_quality_report_yields_no_date_range_not_a_crash():
    metadata = _map({})
    assert metadata["date_range_start"] is None
    assert metadata["date_range_end"] is None


def test_quality_report_with_no_parseable_dates_yields_no_date_range():
    # QualityAssessor itself reports (None, None) when every date value in
    # the dataset was missing/invalid — this must pass through honestly,
    # never fabricated.
    metadata = _map({"quality_report": {"date_range_start": None, "date_range_end": None}})
    assert metadata["date_range_start"] is None
    assert metadata["date_range_end"] is None
