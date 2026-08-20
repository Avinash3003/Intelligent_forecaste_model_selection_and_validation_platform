"""The estimate must be derived from the actual dataset and configuration —
not a fixed placeholder range — and must calibrate from real run history
when enough exists.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from app.config.settings import Settings
from app.orchestration.schemas import JobStatus
from app.schemas.estimation import EstimationRequest
from app.schemas.metadata import MetadataMapping
from app.services.estimation_service import EstimationService, _backtest_window_count


def _frame(groups: int, months_per_group: int = 36) -> pd.DataFrame:
    """A dataset with `groups` keys, each spanning `months_per_group`
    distinct calendar months — matching what `_periods_by_group` measures.
    """
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


class _EmptyHistory:
    """No completed runs at all — forces the heuristic fallback path."""

    def list_runs(self, limit=15):
        return []

    def get_summary(self, run_id):
        return None


class _FakeHistory:
    """A scripted history store with enough usable telemetry to calibrate
    from — enough completed runs, each carrying the timing breakdown this
    phase added.
    """

    def __init__(self, n_runs: int, seconds_per_fit: float = 0.4) -> None:
        self._n_runs = n_runs
        self._seconds_per_fit = seconds_per_fit

    def list_runs(self, limit=15):
        return [
            SimpleNamespace(run_id=f"run-{i}", job_status=JobStatus.COMPLETED)
            for i in range(min(self._n_runs, limit))
        ]

    def get_summary(self, run_id):
        return {
            "evaluation_report": {
                "timing_breakdown": {
                    "backtest_seconds": self._seconds_per_fit * 50,
                    "backtest_windows_evaluated": 50,
                }
            },
            "training_report": {"duration_seconds": self._seconds_per_fit * 10, "trained": 10},
            "explainability_report": {"results": [{}] * 5},
            "stages": [{"name": "Explain Models", "duration_seconds": 2.5}],
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
    return EstimationService(settings=settings or Settings(), history=history or _EmptyHistory())


# ---------------------------------------------------------------------
# Dataset metadata — must be read from the actual file
# ---------------------------------------------------------------------


def test_dataset_summary_reflects_the_real_upload():
    df = _frame(groups=5, months_per_group=24)
    estimate = _service().estimate(df, _request(keys=("store",)))

    assert estimate.dataset.rows == 5 * 24
    assert estimate.dataset.columns == df.shape[1]
    assert estimate.dataset.date_column == "date"
    assert estimate.dataset.target_column == "sales"
    assert estimate.dataset.key_columns == ["store"]
    assert estimate.dataset.unique_keys == 5
    assert estimate.dataset.history_length_periods == 24
    assert estimate.dataset.date_grain in ("Monthly", "Quarterly")  # DateOffset(months=1) spacing


def test_missingness_is_measured_from_the_target_column():
    df = _frame(groups=3, months_per_group=24)
    df.loc[0:5, "sales"] = None
    estimate = _service().estimate(df, _request())
    assert estimate.dataset.missingness_pct is not None
    assert estimate.dataset.missingness_pct > 0


def test_missingness_is_none_when_target_column_is_absent():
    df = _frame(groups=2)
    request = _request()
    request.metadata.target_column = "does_not_exist"
    estimate = _service().estimate(df, request)
    assert estimate.dataset.missingness_pct is None


def test_no_key_columns_means_a_single_series():
    df = _frame(groups=5)
    estimate = _service().estimate(df, _request(keys=()))
    assert estimate.dataset.unique_keys == 1
    assert estimate.workload.forecast_groups == 1


def test_empty_model_selection_estimates_the_whole_registry():
    df = _frame(groups=3)
    estimate = _service().estimate(df, _request(models=()))
    assert estimate.dataset.selected_models  # non-empty
    assert len(estimate.dataset.selected_models) > 1


# ---------------------------------------------------------------------
# Workload — derived from keys x models x real per-key history, not guessed
# ---------------------------------------------------------------------


def test_model_evaluations_is_keys_times_models():
    df = _frame(groups=7, months_per_group=36)
    estimate = _service().estimate(df, _request(models=("lightgbm", "xgboost")))
    assert estimate.workload.model_evaluations == 7 * 2


def test_llm_calls_is_one_per_group_not_per_model():
    # The redesigned LLM engine (Objective 1) calls once per forecast
    # group, never once per (group, model) pair.
    df = _frame(groups=6, months_per_group=36)
    estimate = _service().estimate(df, _request(models=("lightgbm", "xgboost", "arima")))
    assert estimate.workload.llm_calls == 6


def test_backtest_windows_scale_with_history_length():
    short = _service().estimate(_frame(groups=1, months_per_group=18), _request())
    long = _service().estimate(_frame(groups=1, months_per_group=48), _request())
    assert long.workload.backtest_windows > short.workload.backtest_windows


def test_backtest_window_formula_caps_at_max_windows():
    # min_train=12, horizon=3, max_windows=5 -> 5*3=15 additional periods
    # needed past min_train before the cap binds.
    assert _backtest_window_count(history_periods=12 + 3) == 1
    assert _backtest_window_count(history_periods=1000) == 5
    assert _backtest_window_count(history_periods=10) == 0  # below min_train + horizon


def test_tuning_eligible_pairs_respects_the_48_period_threshold():
    df = _frame(groups=4, months_per_group=36)  # below 48
    estimate = _service().estimate(df, _request(models=("lightgbm",)))
    assert estimate.workload.tuning_eligible_pairs == 0

    df2 = _frame(groups=4, months_per_group=60)  # above 48
    estimate2 = _service().estimate(df2, _request(models=("lightgbm",)))
    assert estimate2.workload.tuning_eligible_pairs == 4


def test_seasonal_naive_is_never_tuning_eligible():
    df = _frame(groups=3, months_per_group=60)
    estimate = _service().estimate(df, _request(models=("seasonal_naive",)))
    assert estimate.workload.tuning_eligible_pairs == 0


def test_tft_is_excluded_from_forward_validation_below_its_history_bar():
    # tft needs 60 observations; a 36-month dataset never reaches training.
    df = _frame(groups=5, months_per_group=36)
    estimate = _service().estimate(df, _request(models=("tft", "lightgbm")))
    # Only lightgbm's 5 pairs are eligible; tft's 5 are not.
    assert estimate.workload.forward_validation_forecasts == 5


# ---------------------------------------------------------------------
# Runtime — a real range, scaling with real workload
# ---------------------------------------------------------------------


def test_the_range_is_a_real_range():
    estimate = _service().estimate(_frame(groups=10, months_per_group=36), _request())
    assert estimate.estimated_minutes_low < estimate.estimated_minutes_high
    assert estimate.estimated_minutes_low > 0


def test_more_groups_take_longer():
    small = _service().estimate(_frame(groups=3, months_per_group=36), _request())
    large = _service().estimate(_frame(groups=40, months_per_group=36), _request())
    assert large.estimated_minutes_high > small.estimated_minutes_high


def test_a_heavier_model_mix_takes_longer():
    df = _frame(groups=10, months_per_group=36)
    light = _service().estimate(df, _request(models=("seasonal_naive",)))
    # Prophet, not TFT: TFT is offered but never executed, so it is stripped
    # from the workload and would estimate as no work at all.
    heavy = _service().estimate(df, _request(models=("prophet",)))
    assert heavy.estimated_minutes_high > light.estimated_minutes_high


def test_databricks_execution_adds_cluster_startup_time():
    df = _frame(groups=3, months_per_group=36)
    local = _service(Settings(execution_mode="local")).estimate(df, _request())
    cloud = _service(Settings(execution_mode="databricks")).estimate(df, _request())
    assert cloud.estimated_minutes_low > local.estimated_minutes_low
    assert any("startup" in item.label.lower() for item in cloud.breakdown)



def test_databricks_cost_is_unavailable_without_a_configured_rate():
    df = _frame(groups=5, months_per_group=36)
    settings = Settings(execution_mode="databricks", compute_cost_per_hour=None)
    estimate = _service(settings).estimate(df, _request())
    assert estimate.cost.databricks_cost_available is False
    assert estimate.cost.databricks_cost_low is None


def test_databricks_cost_is_computed_when_rate_is_configured():
    df = _frame(groups=10, months_per_group=36)
    settings = Settings(execution_mode="databricks", compute_cost_per_hour=8.0)
    estimate = _service(settings).estimate(df, _request())
    assert estimate.cost.databricks_cost_available is True
    assert 0 < estimate.cost.databricks_cost_low <= estimate.cost.databricks_cost_high


def test_databricks_cost_is_not_charged_for_local_execution():
    df = _frame(groups=5, months_per_group=36)
    settings = Settings(execution_mode="local", compute_cost_per_hour=8.0)
    estimate = _service(settings).estimate(df, _request())
    assert estimate.cost.databricks_cost_available is False



def test_llm_cost_is_unavailable_without_configured_pricing():
    df = _frame(groups=5, months_per_group=36)
    estimate = _service(Settings()).estimate(df, _request())
    assert estimate.cost.llm_cost_available is False


def test_llm_cost_scales_with_group_count_when_priced():
    settings = Settings(azure_openai_price_input_per_1k=0.15, azure_openai_price_output_per_1k=0.6)
    small = _service(settings).estimate(_frame(groups=3, months_per_group=36), _request())
    large = _service(settings).estimate(_frame(groups=30, months_per_group=36), _request())
    assert small.cost.llm_cost_available and large.cost.llm_cost_available
    assert large.cost.llm_cost_high > small.cost.llm_cost_high


def test_total_cost_sums_both_components():
    settings = Settings(
        execution_mode="databricks", compute_cost_per_hour=8.0,
        azure_openai_price_input_per_1k=0.15, azure_openai_price_output_per_1k=0.6,
    )
    estimate = _service(settings).estimate(_frame(groups=10, months_per_group=36), _request())
    assert estimate.cost.total_cost_available is True
    assert estimate.cost.total_cost_low == pytest.approx(
        estimate.cost.databricks_cost_low + estimate.cost.llm_cost_low, abs=0.01
    )


def test_no_pricing_at_all_means_no_total_cost():
    df = _frame(groups=5, months_per_group=36)
    estimate = _service(Settings(execution_mode="local")).estimate(df, _request())
    assert estimate.cost.total_cost_available is False
    assert estimate.cost.total_cost_low is None


# ---------------------------------------------------------------------
# Historical calibration
# ---------------------------------------------------------------------


def test_insufficient_history_falls_back_to_the_heuristic():
    df = _frame(groups=5, months_per_group=36)
    estimate = _service(history=_EmptyHistory()).estimate(df, _request())
    assert "heuristic" in estimate.calibration_basis


def test_sufficient_history_calibrates_from_it():
    df = _frame(groups=5, months_per_group=36)
    estimate = _service(history=_FakeHistory(n_runs=5)).estimate(df, _request())
    assert "historical" in estimate.calibration_basis
    assert "5" in estimate.calibration_basis


def test_calibration_requires_the_minimum_run_count():
    df = _frame(groups=5, months_per_group=36)
    # Below _MIN_RUNS_FOR_CALIBRATION (3) — still heuristic.
    estimate = _service(history=_FakeHistory(n_runs=2)).estimate(df, _request())
    assert "heuristic" in estimate.calibration_basis


def test_a_broken_history_store_degrades_to_heuristic_not_a_crash():
    class _BrokenHistory:
        def list_runs(self, limit=15):
            raise RuntimeError("MLflow unreachable")

    df = _frame(groups=5, months_per_group=36)
    estimate = _service(history=_BrokenHistory()).estimate(df, _request())
    assert "heuristic" in estimate.calibration_basis
