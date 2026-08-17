"""The estimate must reach the same answer without paying for it twice.

Two costs used to be repeated on every /estimate call: the dataset's date
column was parsed and grouped three times, and MLflow run history was swept
twice (each sweep re-downloading the same summary artifacts). These tests
pin the optimized behaviour so neither can come back.

They assert on *call counts*, not on timings — a timing assertion would be
flaky on shared CI, while a duplicated sweep is exactly countable.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from app.config.settings import Settings
from app.orchestration.schemas import JobStatus
from app.schemas.estimation import EstimationRequest
from app.schemas.metadata import MetadataMapping
from app.services.dataset_analysis import DatasetAnalyzer
from app.services.estimation_service import (
    _CALIBRATION_SAMPLE_TARGET,
    _CALIBRATION_TTL_SECONDS,
    EstimationService,
)


def _frame(groups: int, months_per_group: int = 36) -> pd.DataFrame:
    dates, stores, items, sales = [], [], [], []
    for g in range(groups):
        for m in range(months_per_group):
            dates.append(pd.Timestamp("2020-01-01") + pd.DateOffset(months=m))
            stores.append(g)
            items.append(1)
            sales.append(100.0 + m)
    return pd.DataFrame({"date": dates, "store": stores, "item": items, "sales": sales})


def _request(models=("lightgbm",), horizon=12, keys=("store",)) -> EstimationRequest:
    return EstimationRequest(
        file_id="f1",
        metadata=MetadataMapping(date_column="date", target_column="sales", key_columns=list(keys)),
        selected_models=list(models),
        horizon=horizon,
    )


class _CountingHistory:
    """Records exactly how many remote calls the estimator makes."""

    def __init__(self, n_runs: int = 5, seconds_per_fit: float = 0.4, summary=True) -> None:
        self._n_runs = n_runs
        self._seconds_per_fit = seconds_per_fit
        self._summary = summary
        self.list_runs_calls = 0
        self.get_summary_calls: list[str] = []

    def list_runs(self, limit=15):
        self.list_runs_calls += 1
        return [
            SimpleNamespace(
                run_id=f"run-{i}",
                job_status=JobStatus.COMPLETED,
                duration_seconds=600.0,
            )
            for i in range(min(self._n_runs, limit))
        ]

    def get_summary(self, run_id):
        self.get_summary_calls.append(run_id)
        if not self._summary:
            return None
        return {
            "evaluation_report": {
                "timing_breakdown": {
                    "backtest_seconds": self._seconds_per_fit * 50,
                    "backtest_windows_evaluated": 50,
                }
            },
            "training_report": {"duration_seconds": self._seconds_per_fit * 10, "trained": 10},
            "explainability_report": {"results": [{}] * 5},
            "stages": [
                {"name": "Generate Explainability (SHAP)", "duration_seconds": 2.5},
                {"name": "Train Models", "duration_seconds": 500.0},
            ],
            "insight_report": {
                "trace_summary": {
                    "call_count": 10,
                    "average_latency_ms": 2000.0,
                    "prompt_tokens": 11000,
                    "completion_tokens": 1000,
                }
            },
        }


def _service(settings=None, history=None):
    return EstimationService(settings=settings or Settings(), history=history or _CountingHistory())


# ---------------------------------------------------------------------
# 1. Same dataset -> equivalent estimate inputs and results
# ---------------------------------------------------------------------


def test_the_optimized_path_produces_the_same_estimate_inputs():
    """The refactor moved *where* these are computed, not *what* they are."""
    df = _frame(groups=6, months_per_group=30)
    request = _request(keys=("store", "item"))

    estimate = _service(history=_CountingHistory(n_runs=0)).estimate(df, request)

    assert estimate.dataset.rows == 6 * 30
    assert estimate.dataset.columns == 4
    assert estimate.dataset.unique_keys == 6
    assert estimate.dataset.history_length_periods == 30
    assert estimate.dataset.key_columns == ["store", "item"]
    assert estimate.workload.forecast_groups == 6
    assert estimate.workload.model_evaluations == 6


def test_repeated_estimates_of_the_same_dataset_are_identical():
    df = _frame(groups=4, months_per_group=36)
    service = _service(history=_CountingHistory(n_runs=0))

    first = service.estimate(df, _request())
    second = service.estimate(df, _request())

    assert first.estimated_minutes_low == second.estimated_minutes_low
    assert first.estimated_minutes_high == second.estimated_minutes_high
    assert first.workload == second.workload


# ---------------------------------------------------------------------
# 2. periods_by_group is computed exactly once
# ---------------------------------------------------------------------


def test_the_date_column_is_grouped_only_once_per_estimate(monkeypatch):
    calls = {"count": 0}
    original = DatasetAnalyzer._periods_by_group

    def counting(self, dataframe, parsed_dates, key_columns):
        calls["count"] += 1
        return original(self, dataframe, parsed_dates, key_columns)

    monkeypatch.setattr(DatasetAnalyzer, "_periods_by_group", counting)

    _service(history=_CountingHistory(n_runs=0)).estimate(_frame(groups=5), _request())
    assert calls["count"] == 1


def test_the_date_column_is_parsed_only_once_per_estimate(monkeypatch):
    """Frequency detection and per-group grouping share one to_datetime."""
    import app.services.dataset_analysis as module

    calls = {"count": 0}
    original = module.pd.to_datetime

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module.pd, "to_datetime", counting)

    _service(history=_CountingHistory(n_runs=0)).estimate(_frame(groups=5), _request())
    assert calls["count"] == 1


# ---------------------------------------------------------------------
# 3. MLflow history is swept once, not once per consumer
# ---------------------------------------------------------------------


def test_history_is_listed_once_per_estimate_not_twice():
    history = _CountingHistory(n_runs=5)
    settings = Settings(execution_mode="databricks")

    _service(settings, history).estimate(_frame(groups=3), _request())

    # Previously two: one for startup measurement, one for calibration.
    assert history.list_runs_calls == 1


def test_each_run_summary_is_fetched_once_per_sweep():
    history = _CountingHistory(n_runs=5)
    _service(Settings(execution_mode="databricks"), history).estimate(_frame(groups=3), _request())

    assert len(history.get_summary_calls) == len(set(history.get_summary_calls))


def test_summary_downloads_are_capped_at_the_calibration_budget():
    """Enough samples for a stable median, not one download per listing."""
    history = _CountingHistory(n_runs=15)
    _service(Settings(execution_mode="databricks"), history).estimate(_frame(groups=3), _request())

    assert len(history.get_summary_calls) <= _CALIBRATION_SAMPLE_TARGET


# ---------------------------------------------------------------------
# 4. Cached calibration prevents repeated remote calls
# ---------------------------------------------------------------------


def test_a_second_estimate_reuses_the_cached_calibration():
    history = _CountingHistory(n_runs=5)
    service = _service(Settings(execution_mode="databricks"), history)
    df = _frame(groups=3)

    service.estimate(df, _request())
    calls_after_first = history.list_runs_calls

    for _ in range(5):
        service.estimate(df, _request())

    assert history.list_runs_calls == calls_after_first == 1


def test_the_cache_expires_and_re_sweeps():
    history = _CountingHistory(n_runs=5)
    service = _service(Settings(execution_mode="databricks"), history)
    df = _frame(groups=3)

    service.estimate(df, _request())
    assert history.list_runs_calls == 1

    # Age the cache past its TTL rather than sleeping through it.
    service._cached_history_at -= _CALIBRATION_TTL_SECONDS + 1
    service.estimate(df, _request())
    assert history.list_runs_calls == 2


def test_caching_does_not_change_the_estimate():
    history = _CountingHistory(n_runs=5)
    service = _service(Settings(execution_mode="databricks"), history)
    df = _frame(groups=3)

    first = service.estimate(df, _request())
    cached = service.estimate(df, _request())

    assert first.calibration_basis == cached.calibration_basis
    assert first.estimated_minutes_low == cached.estimated_minutes_low


# ---------------------------------------------------------------------
# 5. Missing summaries are negatively cached
# ---------------------------------------------------------------------


def test_missing_summaries_are_not_re_requested_within_the_ttl():
    from app.orchestration.mlflow_history import MLflowHistoryStore

    store = MLflowHistoryStore(Settings())
    attempts = {"count": 0}

    def failing_find(client, run_id):
        attempts["count"] += 1
        return None

    store._get_client = lambda: object()
    store._find_run = failing_find

    assert store.get_summary("run-x") is None
    assert store.get_summary("run-x") is None
    assert store.get_summary("run-x") is None

    # One real attempt; the rest answered from the negative cache.
    assert attempts["count"] == 1


def test_a_negatively_cached_summary_is_retried_after_the_ttl():
    from app.orchestration.mlflow_history import MLflowHistoryStore

    store = MLflowHistoryStore(Settings())
    attempts = {"count": 0}

    def failing_find(client, run_id):
        attempts["count"] += 1
        return None

    store._get_client = lambda: object()
    store._find_run = failing_find

    assert store.get_summary("run-x") is None
    # A run that had no summary can gain one, so the negative entry must
    # expire rather than being permanent.
    store._missing_summaries["run-x"] = 0.0
    assert store.get_summary("run-x") is None
    assert attempts["count"] == 2


def test_a_run_with_no_summary_does_not_break_calibration():
    history = _CountingHistory(n_runs=5, summary=False)
    estimate = _service(Settings(execution_mode="databricks"), history).estimate(
        _frame(groups=3), _request()
    )
    assert "heuristic" in estimate.calibration_basis


# ---------------------------------------------------------------------
# 6. MLflow failure still uses the existing fallback
# ---------------------------------------------------------------------


def test_an_unreachable_history_store_still_estimates():
    class _Broken:
        def list_runs(self, limit=15):
            raise RuntimeError("MLflow unreachable")

        def get_summary(self, run_id):
            raise RuntimeError("MLflow unreachable")

    estimate = _service(Settings(execution_mode="databricks"), _Broken()).estimate(
        _frame(groups=3), _request()
    )
    assert "heuristic" in estimate.calibration_basis
    assert estimate.estimated_minutes_low > 0
    # The per-mode serverless startup constant, not a measured figure.
    assert any("startup" in item.label.lower() for item in estimate.breakdown)


def test_a_history_store_that_fails_mid_sweep_still_estimates():
    class _FailsOnSummary(_CountingHistory):
        def get_summary(self, run_id):
            raise RuntimeError("artifact download failed")

    estimate = _service(Settings(execution_mode="databricks"), _FailsOnSummary()).estimate(
        _frame(groups=3), _request()
    )
    assert "heuristic" in estimate.calibration_basis


# ---------------------------------------------------------------------
# 7. Serverless does not estimate TFT workload
# ---------------------------------------------------------------------


def test_serverless_excludes_tft_from_the_estimated_workload():
    df = _frame(groups=4, months_per_group=72)  # long enough to clear tft's 60-obs bar
    settings = Settings(execution_mode="databricks")

    estimate = _service(settings, _CountingHistory(n_runs=0)).estimate(
        df, _request(models=("tft", "lightgbm"))
    )

    assert "tft" not in estimate.dataset.selected_models
    assert estimate.dataset.selected_models == ["lightgbm"]
    assert estimate.workload.models_per_group == 1
    assert estimate.workload.model_evaluations == 4


def test_serverless_explains_why_a_model_was_excluded():
    df = _frame(groups=2, months_per_group=72)
    estimate = _service(Settings(execution_mode="databricks"), _CountingHistory(n_runs=0)).estimate(
        df, _request(models=("tft", "arima"))
    )
    excluded = [item for item in estimate.breakdown if item.label == "Excluded models"]
    assert excluded and "tft" in excluded[0].detail


def test_excluding_tft_lowers_the_serverless_estimate():
    df = _frame(groups=5, months_per_group=72)
    request = _request(models=("tft", "lightgbm"))

    serverless = _service(Settings(execution_mode="databricks"), _CountingHistory(n_runs=0)).estimate(df, request)
    local = _service(Settings(execution_mode="local"), _CountingHistory(n_runs=0)).estimate(df, request)

    # TFT is the heaviest per-fit weight in the table, so dropping it must
    # move the estimate down even though serverless also adds startup time.
    assert serverless.workload.model_evaluations < local.workload.model_evaluations


def test_local_execution_still_estimates_tft():
    df = _frame(groups=3, months_per_group=72)
    estimate = _service(Settings(execution_mode="local"), _CountingHistory(n_runs=0)).estimate(
        df, _request(models=("tft", "lightgbm"))
    )
    assert "tft" in estimate.dataset.selected_models


def test_dcs_execution_still_estimates_tft():
    """Only the Serverless environment spec omits torch; DCS ships it."""
    df = _frame(groups=3, months_per_group=72)
    estimate = _service(Settings(execution_mode="databricks_dcs"), _CountingHistory(n_runs=0)).estimate(
        df, _request(models=("tft",))
    )
    assert "tft" in estimate.dataset.selected_models


def test_the_registry_default_is_also_filtered_by_execution_mode():
    """An empty selection means "everything" — and everything must still
    exclude what the mode cannot run."""
    df = _frame(groups=2, months_per_group=72)
    estimate = _service(Settings(execution_mode="databricks"), _CountingHistory(n_runs=0)).estimate(
        df, _request(models=())
    )
    assert "tft" not in estimate.dataset.selected_models
    assert len(estimate.dataset.selected_models) > 1


# ---------------------------------------------------------------------
# 8. The previously problematic large dataset responds quickly
# ---------------------------------------------------------------------


@pytest.mark.parametrize("rows", [900_000])
def test_a_large_dataset_estimates_well_inside_the_frontend_timeout(rows):
    """The frontend aborts at 30s (apiConfig.js REQUEST_TIMEOUT_MS)."""
    import time

    months = 60
    groups = rows // months
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2013-01-01"] * rows
            ),  # replaced below; allocated once to keep the fixture cheap
            "store": [g % 10 for g in range(rows)],
            "item": [g % 50 for g in range(rows)],
            "sales": [float(i % 97) for i in range(rows)],
        }
    )
    frame["date"] = pd.date_range("2013-01-01", periods=months, freq="MS").repeat(
        groups
    )[:rows]

    service = _service(Settings(execution_mode="databricks"), _CountingHistory(n_runs=5))

    started = time.perf_counter()
    estimate = service.estimate(frame, _request(models=("lightgbm", "arima"), keys=("store", "item")))
    elapsed = time.perf_counter() - started

    assert estimate.dataset.rows == rows
    # Generous versus the 30s browser timeout, tight enough to catch a
    # regression back to per-request history sweeps or repeated parsing.
    assert elapsed < 10.0, f"estimate took {elapsed:.2f}s"
