"""FlatForecast, ExcessiveSmoothing and MissingSeasonality also had to move
onto the reference window - confirmed by real-run evidence, not assumption.

FlatForecast/ExcessiveSmoothing compare a VARIABILITY SCALE (forecast std vs
history std). On a series that changed level, full-history std is inflated
by the level shift itself, so a forecast correctly tracking the current
regime's own variation reads as artificially "flat". Real evidence from the
stored gold run (mlruns/a3daecdbe226439ab1a49ee3d3949a60): LightGBM (WMAPE
4.00%, the run's most accurate model) and XGBoost (5.21%) were eliminated
for Excessive Smoothing at full-history ratios of 0.135/0.106 against a 0.15
floor; against the window they are 0.210/0.165 and survive - while ARIMA,
genuinely near-flat at 0.035, still fails.

MissingSeasonality's GATE is an autocorrelation on non-detrended data, which
conflates trend with seasonality: any sustained trend lifts autocorrelation
at every lag. Gold (no annual seasonality) scored 0.707 on full history vs
0.004 on the window; genuinely seasonal series stay detected (airline
passengers 0.486, Time Series Practice 0.661 on the window, both above the
0.3 gate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_engine.config.evaluation_config import ForwardValidationConfig
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.s06_evaluation.evaluation_report import ForwardForecast
from forecast_engine.s06_evaluation.forward_validator import ForwardForecastValidator
from forecast_engine.s06_evaluation.validation_rules import (
    ExcessiveSmoothingRule,
    FlatForecastRule,
    MissingSeasonalityRule,
)


def _series(values, frequency: str = "Monthly") -> ForecastSeries:
    frame = pd.DataFrame(
        {"date": pd.date_range("2000-01-01", periods=len(values), freq="MS"), "target": values}
    )
    return ForecastSeries(
        group_id="g", frame=frame, date_column="date", target_column="target", frequency=frequency
    )


def _validate(series, values):
    return ForwardForecastValidator(ForwardValidationConfig()).validate(
        series, ForwardForecast(dates=["x"] * len(values), values=values)
    )


def _outcome(result, rule_id):
    return next(o for o in result.rule_outcomes if o.rule_id == rule_id)


# 10 years at ~20 then 3 years at ~100: full-history std is dominated by the
# level shift, recent-window std reflects the current regime's real movement.
_REGIME_CHANGE = [20.0 + (i % 11) * 0.3 for i in range(120)] + [100.0 + (i % 13) * 0.4 for i in range(36)]


def test_the_three_rules_are_marked_as_windowed():
    assert FlatForecastRule.uses_reference_window is True
    assert ExcessiveSmoothingRule.uses_reference_window is True
    assert MissingSeasonalityRule.uses_reference_window is True


def test_a_forecast_tracking_the_current_regime_is_not_called_flat_or_smoothed():
    series = _series(_REGIME_CHANGE)
    forecast = [100.0 + (i % 13) * 0.4 for i in range(12)]

    result = _validate(series, forecast)

    assert _outcome(result, "flat_forecast").passed is True
    assert _outcome(result, "excessive_smoothing").passed is True


def test_a_genuinely_flat_forecast_still_fails_both_rules():
    """The window must not launder real underfitting: a constant forecast
    has ~zero std, so the ratio stays ~zero whichever denominator is used."""
    series = _series(_REGIME_CHANGE)
    forecast = [100.0] * 12

    result = _validate(series, forecast)

    assert _outcome(result, "flat_forecast").passed is False
    assert _outcome(result, "excessive_smoothing").passed is False
    assert result.passed is False


def test_a_heavily_compressed_forecast_still_fails_excessive_smoothing():
    """Retains some shape but almost no amplitude - exactly what the
    excessive-smoothing rule exists to catch, still caught after windowing."""
    series = _series(_REGIME_CHANGE)
    forecast = [100.0 + (i % 13) * 0.005 for i in range(12)]  # ~1% of recent amplitude

    result = _validate(series, forecast)

    assert _outcome(result, "excessive_smoothing").passed is False


def test_a_trending_non_seasonal_series_is_not_treated_as_seasonal():
    """Trend-induced autocorrelation must not impose a seasonality
    obligation on a series that has no seasonality."""
    trending = [20.0 + i * 0.8 for i in range(150)]  # pure trend, no seasonal cycle
    series = _series(trending)
    forecast = [140.0 + i * 0.8 for i in range(12)]  # continues the trend, no swing

    result = _validate(series, forecast)

    assert _outcome(result, "missing_seasonality").passed is True


def test_a_genuinely_seasonal_series_still_requires_seasonality():
    """A real seasonal cycle must still be detected through the window, and
    a forecast that drops it must still be caught."""
    seasonal = [100.0 + 40.0 * np.sin(2 * np.pi * i / 12) for i in range(150)]
    series = _series(seasonal)
    flat_forecast = [100.0 + (i % 2) * 0.2 for i in range(12)]  # seasonality dropped

    result = _validate(series, flat_forecast)

    assert _outcome(result, "missing_seasonality").passed is False


def test_a_seasonal_forecast_on_a_seasonal_series_passes():
    seasonal = [100.0 + 40.0 * np.sin(2 * np.pi * i / 12) for i in range(150)]
    series = _series(seasonal)
    good_forecast = [100.0 + 40.0 * np.sin(2 * np.pi * (150 + i) / 12) for i in range(12)]

    result = _validate(series, good_forecast)

    assert _outcome(result, "missing_seasonality").passed is True


def test_stable_series_behaviour_is_unchanged_by_the_windowing():
    """On a series with no regime change, the window and the full history
    describe the same variability, so these rules behave identically."""
    stable = [50.0 + (i % 7) for i in range(150)]
    series = _series(stable)
    forecast = [50.0 + (i % 7) for i in range(12)]

    result = _validate(series, forecast)

    assert _outcome(result, "flat_forecast").passed is True
    assert _outcome(result, "excessive_smoothing").passed is True


def test_constant_history_still_short_circuits_safely():
    """Zero-variance history has no scale to compare against; the existing
    `historical_std == 0` guard must still fire on the windowed slice."""
    series = _series([50.0] * 60)
    result = _validate(series, [50.0] * 12)

    assert _outcome(result, "flat_forecast").passed is True
    assert _outcome(result, "excessive_smoothing").passed is True
