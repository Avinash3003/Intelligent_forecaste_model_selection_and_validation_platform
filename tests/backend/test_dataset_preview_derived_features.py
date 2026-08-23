"""Priority C, Step 5 — the curated-data preview reflects the run's ACTUAL
derived feature selection: real computed columns, grouped correctly by
business key, never a fake/static list.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.orchestration.schemas import ExecutionBackend, JobStatus, PipelineExecutionResult
from app.services.dataset_preview_service import DatasetPreviewService


@dataclass
class _FakeExecutor:
    curated_uri: str
    derived_features: list | None
    key_columns: list

    def get_result(self, run_id: str) -> PipelineExecutionResult:
        return PipelineExecutionResult(
            run_id=run_id,
            job_status=JobStatus.COMPLETED,
            execution_backend=ExecutionBackend.LOCAL,
            run_metadata={
                "curated_dataset_uri": self.curated_uri,
                "configuration": {
                    "date_column": "date",
                    "target_column": "sales",
                    "key_columns": self.key_columns,
                },
                "derived_features": self.derived_features,
            },
        )


def _write_curated(tmp_path, rows: list[str], header: str = "date,store,sales") -> str:
    path = tmp_path / "curated.csv"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


def test_selected_features_are_added_as_real_columns(tmp_path):
    rows = [f"2020-01-{d:02d},1,{d}" for d in range(1, 15)]
    uri = _write_curated(tmp_path, rows)
    service = DatasetPreviewService(executor=_FakeExecutor(uri, ["lag_1", "month"], ["store"]))

    preview = service.get_preview("run-1")

    assert "lag_1" in preview.columns
    assert "month" in preview.columns
    assert "lag_2" not in preview.columns
    assert "rolling_mean_3" not in preview.columns


def test_lag_values_are_computed_correctly_within_one_group(tmp_path):
    rows = [f"2020-01-{d:02d},1,{d * 10}" for d in range(1, 6)]  # sales: 10,20,30,40,50
    uri = _write_curated(tmp_path, rows)
    service = DatasetPreviewService(executor=_FakeExecutor(uri, ["lag_1"], ["store"]))

    preview = service.get_preview("run-1")
    lag_idx = preview.columns.index("lag_1")
    lag_values = [row[lag_idx] for row in preview.rows]

    # Row 1 has no prior value; rows 2..5 lag the previous day's sales.
    assert lag_values == ["", "10.0", "20.0", "30.0", "40.0"]


def test_lags_never_cross_between_two_different_business_keys(tmp_path):
    rows = [
        "2020-01-01,1,100",
        "2020-01-02,1,200",
        "2020-01-01,2,900",
        "2020-01-02,2,950",
    ]
    uri = _write_curated(tmp_path, rows)
    service = DatasetPreviewService(executor=_FakeExecutor(uri, ["lag_1"], ["store"]))

    preview = service.get_preview("run-1")
    store_idx = preview.columns.index("store")
    lag_idx = preview.columns.index("lag_1")

    for row in preview.rows:
        if row[store_idx] == "2" and row[0] == "2020-01-02":
            # Must lag store 2's own prior value (900), never store 1's.
            assert row[lag_idx] == "900.0"


def test_rolling_mean_is_computed_within_one_group(tmp_path):
    rows = [f"2020-01-{d:02d},1,{v}" for d, v in zip(range(1, 6), [10, 20, 30, 40, 50])]
    uri = _write_curated(tmp_path, rows)
    service = DatasetPreviewService(executor=_FakeExecutor(uri, ["rolling_mean_3"], ["store"]))

    preview = service.get_preview("run-1")
    idx = preview.columns.index("rolling_mean_3")
    values = [row[idx] for row in preview.rows]

    # Trailing 3-obs mean of values *before* the current row: rows 1-3 are
    # incomplete (""), row 4 -> mean(10,20,30)=20, row 5 -> mean(20,30,40)=30.
    assert values[:3] == ["", "", ""]
    assert values[3] == "20.0"
    assert values[4] == "30.0"


def test_no_selection_recorded_means_every_default_feature_is_shown(tmp_path):
    rows = [f"2020-01-{d:02d},1,{d}" for d in range(1, 15)]
    uri = _write_curated(tmp_path, rows)
    service = DatasetPreviewService(executor=_FakeExecutor(uri, None, ["store"]))

    preview = service.get_preview("run-1")

    for expected in ("lag_1", "lag_2", "lag_3", "rolling_mean_3", "rolling_mean_6", "month", "quarter"):
        assert expected in preview.columns


def test_an_explicit_empty_selection_adds_no_derived_columns(tmp_path):
    rows = [f"2020-01-{d:02d},1,{d}" for d in range(1, 15)]
    uri = _write_curated(tmp_path, rows)
    service = DatasetPreviewService(executor=_FakeExecutor(uri, [], ["store"]))

    preview = service.get_preview("run-1")

    assert preview.columns == ["date", "store", "sales"]


def test_base_columns_and_row_values_are_untouched(tmp_path):
    rows = ["2020-01-01,1,42"]
    uri = _write_curated(tmp_path, rows)
    service = DatasetPreviewService(executor=_FakeExecutor(uri, ["month"], ["store"]))

    preview = service.get_preview("run-1")

    assert preview.columns[:3] == ["date", "store", "sales"]
    assert preview.rows[0][:3] == ["2020-01-01", "1", "42"]
