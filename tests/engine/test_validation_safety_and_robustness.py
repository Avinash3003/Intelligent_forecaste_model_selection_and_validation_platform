"""End-to-end safety and numerical-robustness properties of the validation
+ drift chain, after the reference-window, PSI and underfitting-rule fixes.

These are the guard-rail tests: the whole point of the reference-window work
was to stop rejecting GOOD forecasts, and this file pins the other half of
that contract - genuinely unsafe forecasts must still be rejected, and no
degenerate input may crash, hang, or produce a non-finite statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_engine.config.drift_config import DriftValidationConfig
from forecast_engine.config.evaluation_config import ForwardValidationConfig
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.s06_evaluation.evaluation_report import ForwardForecast
from forecast_engine.s06_evaluation.forward_validator import ForwardForecastValidator
from forecast_engine.s09_drift.drift_validator import DriftValidator


def _series(values, frequency: str = "Monthly") -> ForecastSeries:
    frame = pd.DataFrame(
        {"date": pd.date_range("2000-01-01", periods=len(values), freq="MS"), "target": values}
    )
    return ForecastSeries(
        group_id="g", frame=frame, date_column="date", target_column="target", frequency=frequency
    )


def _validate(values, forecast_values, frequency="Monthly"):
    return ForwardForecastValidator(ForwardValidationConfig()).validate(
        _series(values, frequency), ForwardForecast(dates=["x"] * len(forecast_values), values=forecast_values)
    )


# ---------------------------------------------------------------------
# Extreme forecasts must STILL be rejected
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,history,forecast",
    [
        ("history~60 -> forecast~300", [60.0 + (i % 4) for i in range(60)], [300.0] * 12),
        ("history~100 -> forecast~1000", [100.0 + (i % 5) for i in range(60)], [1000.0] * 12),
        ("stable -> 10x larger", [50.0 + (i % 6) for i in range(60)], [500.0 + (i % 6) for i in range(12)]),
        ("regime ~100 -> forecast 500",
         [20.0 + (i % 11) * 0.3 for i in range(120)] + [100.0 + (i % 13) * 0.4 for i in range(36)],
         [500.0] * 12),
        ("collapse to near-zero", [100.0 + (i % 7) for i in range(60)], [0.5] * 12),
    ],
)
def test_extreme_forecasts_are_still_eliminated(label, history, forecast):
    result = _validate(history, forecast)
    assert result.passed is False, f"{label} should have been eliminated"


def test_abnormal_growth_remains_a_hard_gate_on_the_canonical_case():
    """Explicitly pinned: AbnormalGrowthRule must stay hard and must stay
    the rule that catches a level jump, independent of the windowing work."""
    result = _validate([60.0 + (i % 4) for i in range(60)], [300.0] * 12)
    growth = next(o for o in result.rule_outcomes if o.rule_id == "abnormal_growth")
    assert growth.passed is False


def test_non_finite_forecast_remains_a_hard_failure():
    for bad in (float("nan"), float("inf"), float("-inf")):
        result = _validate([50.0 + (i % 5) for i in range(60)], [bad] * 12)
        finite = next(o for o in result.rule_outcomes if o.rule_id == "non_finite_forecast")
        assert finite.passed is False
        assert result.passed is False


# ---------------------------------------------------------------------
# Legitimate forecasts must SURVIVE
# ---------------------------------------------------------------------


def test_a_legitimate_regime_change_continuation_survives():
    """The design review's own example: a series that ramped 20 -> 80 and a
    forecast continuing at 85-98 must not fail merely for sitting above the
    OLD range."""
    ramp = [20, 22, 25, 28, 30, 32, 35, 38, 40, 45, 50, 55, 60, 65, 70, 72, 75, 78,
            80, 82, 79, 81, 83, 80, 82, 84, 81, 83, 85, 82, 84, 86, 83, 85, 87, 84]
    forecast = [85.0, 87.0, 90.0, 92.0, 95.0, 93.0, 96.0, 94.0, 97.0, 95.0, 98.0, 96.0]

    result = _validate([float(v) for v in ramp], forecast)

    assert result.passed is True, [o.rule_name for o in result.rule_outcomes if not o.passed]


def test_a_forecast_continuing_a_new_plateau_survives():
    regime = [20.0 + (i % 11) * 0.3 for i in range(120)] + [100.0 + (i % 13) * 0.4 for i in range(36)]
    result = _validate(regime, [100.0 + (i % 13) * 0.4 for i in range(12)])
    assert result.passed is True, [o.rule_name for o in result.rule_outcomes if not o.passed]


# ---------------------------------------------------------------------
# Numerical robustness - nothing may crash or return a non-finite stat
# ---------------------------------------------------------------------


_DEGENERATE_CASES = [
    ("constant history, constant forecast", [50.0] * 40, [50.0] * 12),
    ("constant history, varying forecast", [50.0] * 40, [50.0 + i for i in range(12)]),
    ("all-zero history", [0.0] * 40, [0.0] * 12),
    ("single-point forecast", [50.0 + i % 5 for i in range(40)], [52.0]),
    ("negative values", [-50.0 - i % 5 for i in range(40)], [-52.0] * 12),
    ("huge magnitudes", [1e12 + i for i in range(40)], [1e12 + 5] * 12),
    ("tiny magnitudes", [1e-12 * (1 + i % 3) for i in range(40)], [1e-12] * 12),
    ("crossing zero", [float(i - 20) for i in range(40)], [1.0] * 12),
]


@pytest.mark.parametrize("label,history,forecast", _DEGENERATE_CASES)
def test_forward_validation_never_crashes_on_degenerate_input(label, history, forecast):
    result = _validate(history, forecast)
    assert isinstance(result.passed, bool)
    for outcome in result.rule_outcomes:
        for value in outcome.measurements.values():
            if isinstance(value, float):
                assert not np.isnan(value), f"{label}: {outcome.rule_id} produced NaN"


@pytest.mark.parametrize("label,history,forecast", _DEGENERATE_CASES)
def test_drift_never_produces_a_non_finite_statistic(label, history, forecast):
    validator = DriftValidator(DriftValidationConfig.default())
    result = validator.validate(
        _series(history), ForwardForecast(dates=["x"] * len(forecast), values=forecast)
    )
    assert np.isfinite(result.drift_statistic), f"{label}: non-finite drift statistic"
    assert np.isfinite(result.threshold.value)
    assert result.threshold.value >= 0.0


def test_drift_raises_explicitly_rather_than_scoring_a_non_finite_forecast():
    """An all-NaN forecast is not silently scored - drift raises, and
    production_selector catches it per candidate (documented behaviour),
    so one unusable candidate never stops the others being evaluated."""
    validator = DriftValidator(DriftValidationConfig.default())
    with pytest.raises(ValueError):
        validator.validate(
            _series([50.0 + i % 5 for i in range(40)]),
            ForwardForecast(dates=["x"] * 12, values=[float("nan")] * 12),
        )


def test_history_containing_nan_is_filtered_not_propagated():
    history = [float("nan") if i % 7 == 0 else 50.0 + (i % 5) for i in range(60)]
    result = _validate(history, [52.0 + (i % 5) for i in range(12)])
    assert isinstance(result.passed, bool)
    for outcome in result.rule_outcomes:
        for value in outcome.measurements.values():
            if isinstance(value, float):
                assert not np.isnan(value)


def test_short_history_is_skipped_explicitly_not_silently_passed():
    """Below min_history_for_validation the rules would be judging noise;
    that must be reported as an explicit skip reason, not an unexplained
    pass."""
    result = _validate([50.0, 51.0, 52.0, 53.0, 54.0], [55.0] * 3)
    assert result.skipped_reason is not None
    assert "observation" in result.skipped_reason
