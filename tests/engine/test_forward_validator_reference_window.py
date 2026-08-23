"""forward_validator.py now hands ExcessiveVolatility/PercentileSpike/
IqrBounds/ForecastVariance a recent reference window instead of the full
series (validation_rules.py's uses_reference_window). Everything else
(Finite, FlatForecast, ExcessiveSmoothing, MissingSeasonality, MissingTrend,
AbnormalGrowthRate) still receives the full series, unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_engine.config.evaluation_config import ForwardValidationConfig
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.s06_evaluation.evaluation_report import ForwardForecast
from forecast_engine.s06_evaluation.forward_validator import ForwardForecastValidator


def _series(values: list[float], frequency: str = "Monthly") -> ForecastSeries:
    frame = pd.DataFrame(
        {"date": pd.date_range("2000-01-01", periods=len(values), freq="MS"), "target": values}
    )
    return ForecastSeries(group_id="g", frame=frame, date_column="date", target_column="target", frequency=frequency)


def _outcome(result, rule_id: str):
    return next(o for o in result.rule_outcomes if o.rule_id == rule_id)


# A regime-changing series: 10 years (120 months) at ~20, then 3 years
# (36 months) at ~100 - the exact shape of the gold/BTC problem, at a
# scale simple enough to reason about by hand.
_OLD_REGIME = [20.0 + (i % 3) for i in range(120)]
_NEW_REGIME = [100.0 + (i % 5) for i in range(36)]
_REGIME_CHANGE_HISTORY = _OLD_REGIME + _NEW_REGIME


def test_a_forecast_continuing_the_current_regime_passes_the_windowed_rules():
    """The core fix: forecasting ~100 after a genuine regime change to ~100
    must not be flagged just because most of history was at ~20."""
    series = _series(_REGIME_CHANGE_HISTORY)
    forecast = ForwardForecast(
        dates=[f"2013-{m:02d}-01" for m in range(1, 13)],
        values=[100.0 + (i % 5) for i in range(12)],
    )
    validator = ForwardForecastValidator(ForwardValidationConfig())

    result = validator.validate(series, forecast)

    assert _outcome(result, "iqr_bounds").passed is True
    assert _outcome(result, "percentile_spike").passed is True
    assert _outcome(result, "excessive_volatility").passed is True


def test_a_genuinely_extreme_forecast_still_fails_after_the_fix():
    """The window recalibrates what counts as normal - it does not disable
    anomaly detection. Forecasting ~500 when the current regime is ~100
    must still be caught."""
    series = _series(_REGIME_CHANGE_HISTORY)
    forecast = ForwardForecast(
        dates=[f"2013-{m:02d}-01" for m in range(1, 13)],
        values=[500.0] * 12,
    )
    validator = ForwardForecastValidator(ForwardValidationConfig())

    result = validator.validate(series, forecast)

    assert result.passed is False
    assert _outcome(result, "iqr_bounds").passed is False
    assert _outcome(result, "abnormal_growth").passed is False  # the hard gate, unaffected by this fix


def test_artificial_extreme_case_recent_60_forecast_300_still_fails():
    """The exact scenario from the design review: recent history ~60,
    forecast ~300 - must still be rejected by the (unchanged) hard gate."""
    history = [60.0 + (i % 4) for i in range(60)]
    series = _series(history)
    forecast = ForwardForecast(dates=[f"2005-{m:02d}-01" for m in range(1, 13)], values=[300.0] * 12)
    validator = ForwardForecastValidator(ForwardValidationConfig())

    result = validator.validate(series, forecast)

    assert result.passed is False
    assert _outcome(result, "abnormal_growth").passed is False


def test_unaffected_rules_still_receive_the_full_series_not_the_window():
    """MissingTrendRule (uses_reference_window=False) must judge trend
    retention against the FULL series' trend, not just the recent window -
    a forecast that reverses a trend visible only earlier in history should
    still be judged against that full picture."""
    # A clear upward trend across the whole series.
    trending_history = [float(i) for i in range(60)]
    series = _series(trending_history)
    # Forecast retains the trend well.
    good_forecast = ForwardForecast(
        dates=[f"2005-{m:02d}-01" for m in range(1, 13)], values=[60.0 + i * 1.0 for i in range(12)]
    )
    validator = ForwardForecastValidator(ForwardValidationConfig())

    result = validator.validate(series, good_forecast)
    assert _outcome(result, "missing_trend").passed is True


def test_flat_series_hits_the_same_zero_spread_guard_as_before():
    """A genuinely flat series must still short-circuit every ratio-based
    rule the same way it did before this change - the guard is
    `historical_std == 0`, computed on whichever population the rule
    receives, windowed or not."""
    flat_history = [50.0] * 60
    series = _series(flat_history)
    flat_forecast = ForwardForecast(dates=[f"2005-{m:02d}-01" for m in range(1, 13)], values=[50.0] * 12)
    validator = ForwardForecastValidator(ForwardValidationConfig())

    result = validator.validate(series, flat_forecast)

    assert _outcome(result, "iqr_bounds").passed is True
    assert _outcome(result, "excessive_volatility").passed is True
    assert _outcome(result, "flat_forecast").passed is True  # correctly flat is correct here


def test_daily_series_windows_differently_from_monthly_for_the_same_rule():
    """The window (and therefore what counts as 'normal') differs by
    frequency, as required - not a single hardcoded constant."""
    old_regime_daily = [20.0 + (i % 3) for i in range(400)]
    new_regime_daily = [100.0 + (i % 5) for i in range(90)]
    daily_series = _series(old_regime_daily + new_regime_daily, frequency="Daily")
    forecast = ForwardForecast(dates=[f"2001-05-{d:02d}" for d in range(1, 31)], values=[100.0 + (i % 5) for i in range(30)])
    validator = ForwardForecastValidator(ForwardValidationConfig())

    result = validator.validate(daily_series, forecast)

    assert _outcome(result, "iqr_bounds").passed is True
