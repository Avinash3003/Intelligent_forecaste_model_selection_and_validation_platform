"""Priority A — the Results chart's actual-history line reads the FULL
curated dataset (`DatasetPreviewService.get_full_series`), never a bounded
tail, filtered to one business key.
"""

from __future__ import annotations

from dataclasses import dataclass


from app.orchestration.schemas import ExecutionBackend, JobStatus, PipelineExecutionResult
from app.services.dataset_preview_service import DatasetPreviewService


@dataclass
class _FakeExecutor:
    curated_uri: str | None

    def get_result(self, run_id: str) -> PipelineExecutionResult:
        return PipelineExecutionResult(
            run_id=run_id,
            job_status=JobStatus.COMPLETED,
            execution_backend=ExecutionBackend.LOCAL,
            run_metadata={"curated_dataset_uri": self.curated_uri} if self.curated_uri else {},
        )


def _write_curated(tmp_path, rows: list[str]) -> str:
    path = tmp_path / "curated.csv"
    path.write_text("date,store,item,sales\n" + "\n".join(rows) + "\n")
    return str(path)


def test_returns_every_observation_not_a_bounded_tail(tmp_path):
    rows = [f"2020-01-{day:02d},1,1,{day}" for day in range(1, 32)]
    uri = _write_curated(tmp_path, rows)
    service = DatasetPreviewService(executor=_FakeExecutor(uri))

    series = service.get_full_series("run-1", "date", "sales", {"store": 1, "item": 1})

    assert len(series) == 31
    assert series[0] == ("2020-01-01", 1.0)
    assert series[-1] == ("2020-01-31", 31.0)


def test_filters_to_the_requested_business_key_only(tmp_path):
    rows = [
        "2020-01-01,1,1,10",
        "2020-01-01,1,2,999",
        "2020-01-02,1,1,20",
    ]
    uri = _write_curated(tmp_path, rows)
    service = DatasetPreviewService(executor=_FakeExecutor(uri))

    series = service.get_full_series("run-1", "date", "sales", {"store": 1, "item": 1})

    assert series == [("2020-01-01", 10.0), ("2020-01-02", 20.0)]


def test_numeric_key_matches_string_csv_cell(tmp_path):
    # Business keys round-trip through JSON/summary.json as ints/floats
    # while the curated CSV's cells are always strings — must still match.
    uri = _write_curated(tmp_path, ["2020-01-01,1,1,5"])
    service = DatasetPreviewService(executor=_FakeExecutor(uri))

    assert service.get_full_series("run-1", "date", "sales", {"store": 1, "item": 1}) == [("2020-01-01", 5.0)]
    assert service.get_full_series("run-1", "date", "sales", {"store": "1", "item": "1"}) == [("2020-01-01", 5.0)]


def test_no_key_values_means_single_series_returns_every_row(tmp_path):
    uri = _write_curated(tmp_path, ["2020-01-01,1,1,5", "2020-01-02,1,1,6"])
    service = DatasetPreviewService(executor=_FakeExecutor(uri))

    assert service.get_full_series("run-1", "date", "sales", {}) == [("2020-01-01", 5.0), ("2020-01-02", 6.0)]


def test_no_curated_file_returns_none_not_a_crash(tmp_path):
    service = DatasetPreviewService(executor=_FakeExecutor(None))
    assert service.get_full_series("run-1", "date", "sales", {}) is None


def test_unparsable_target_rows_are_skipped_not_a_crash(tmp_path):
    uri = _write_curated(tmp_path, ["2020-01-01,1,1,5", "2020-01-02,1,1,NaN", "2020-01-03,1,1,7"])
    service = DatasetPreviewService(executor=_FakeExecutor(uri))

    assert service.get_full_series("run-1", "date", "sales", {"store": 1, "item": 1}) == [
        ("2020-01-01", 5.0),
        ("2020-01-03", 7.0),
    ]
