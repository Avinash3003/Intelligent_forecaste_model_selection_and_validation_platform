"""Forward-forecast elimination rules (Sections 6.5.1–6.5.3).

Each rule is an independent object that answers one question about a
forecast and returns a structured outcome. Independence is the design
requirement: rules can be enabled or disabled individually, a new signal is
added by registering a class, and one rule failing says nothing about the
others — a model is eliminated if *any* configured rule fails.

Every threshold is relative. Section 6.5.3 requires elimination rules to be
"computed from the key's own history rather than fixed global constants",
so each rule compares the forecast against the same key's historical
variance, percentiles, IQR or trend. That is what allows the identical rule
set to judge a series of unit counts and one of currency amounts.

No narrative text is produced here — only a RejectionReason code and the
measurements behind it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from forecast_engine.config.evaluation_config import (
    OverfittingRulesConfig,
    RejectionReason,
    UnderfittingRulesConfig,
)
from forecast_engine.s06_evaluation.evaluation_report import RuleOutcome


class ValidationRule(ABC):
    """One independent forward-forecast check."""

    rule_id: str = "rule"
    rule_name: str = "Validation Rule"

    # Whether configuration has this rule switched on
    @abstractmethod
    def is_enabled(self) -> bool: ...

    # Judge `forecast` against the key's own `history`
    @abstractmethod
    def evaluate(self, history: np.ndarray, forecast: np.ndarray) -> RuleOutcome: ...

    # Build a passing rule outcome
    def _passed(self, **measurements) -> RuleOutcome:
        return RuleOutcome(
            rule_id=self.rule_id, rule_name=self.rule_name, passed=True, measurements=measurements
        )

    # Build a failing rule outcome
    def _failed(self, reason: RejectionReason, detail: str, **measurements) -> RuleOutcome:
        return RuleOutcome(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            passed=False,
            reason=reason,
            detail=detail,
            measurements=measurements,
        )


# ----------------------------------------------------------------------
# Overfitting signals (Section 6.5.1)
# ----------------------------------------------------------------------


class NonFiniteForecastRule(ValidationRule):
    """Rejects forecasts containing NaN or infinite values.

    Not one of the doc's named signals but a precondition for all of them:
    every other rule's arithmetic is meaningless on non-finite input, and
    such a forecast could never be published.
    """

    rule_id = "non_finite_forecast"
    rule_name = "Finite Forecast Values"

    # Always enabled
    def is_enabled(self) -> bool:
        return True

    # Reject forecasts containing any non-finite value
    def evaluate(self, history: np.ndarray, forecast: np.ndarray) -> RuleOutcome:
        non_finite = int(np.sum(~np.isfinite(forecast)))
        if non_finite:
            return self._failed(
                RejectionReason.NON_FINITE_FORECAST,
                f"{non_finite} forecast value(s) are not finite.",
                non_finite_values=non_finite,
            )
        return self._passed(non_finite_values=0)


class ExcessiveVolatilityRule(ValidationRule):
    """Forecast varies far more than the history it came from."""

    rule_id = "excessive_volatility"
    rule_name = "Excessive Volatility"

    # Store overfitting rule thresholds
    def __init__(self, config: OverfittingRulesConfig) -> None:
        self._config = config

    # Whether volatility-ratio checking is enabled
    def is_enabled(self) -> bool:
        return self._config.volatility_ratio_enabled

    # Reject when forecast volatility swamps historical volatility
    def evaluate(self, history: np.ndarray, forecast: np.ndarray) -> RuleOutcome:
        historical_std = float(np.std(history))
        forecast_std = float(np.std(forecast))

        # A flat history gives no scale to compare against; the flat/
        # smoothing rules cover that case instead.
        if historical_std == 0:
            return self._passed(historical_std=0.0, forecast_std=forecast_std)

        ratio = forecast_std / historical_std
        if ratio > self._config.max_volatility_ratio:
            return self._failed(
                RejectionReason.EXCESSIVE_FORECAST_VOLATILITY,
                f"Forecast volatility is {ratio:.2f}x historical volatility, above the "
                f"{self._config.max_volatility_ratio}x limit.",
                volatility_ratio=ratio,
                threshold=self._config.max_volatility_ratio,
            )
        return self._passed(volatility_ratio=ratio)


class PercentileSpikeRule(ValidationRule):
    """Forecast escapes the historical percentile band."""

    rule_id = "percentile_spike"
    rule_name = "Unrealistic Spikes"

    # Store overfitting rule thresholds
    def __init__(self, config: OverfittingRulesConfig) -> None:
        self._config = config

    # Whether spike detection is enabled
    def is_enabled(self) -> bool:
        return self._config.spike_detection_enabled

    # Reject values that escape the historical percentile band
    def evaluate(self, history: np.ndarray, forecast: np.ndarray) -> RuleOutcome:
        low = float(np.percentile(history, self._config.lower_percentile))
        high = float(np.percentile(history, self._config.upper_percentile))
        spread = high - low

        # Widen the band so a legitimate continuation of trend is not
        # mistaken for a spike.
        tolerance = spread * self._config.percentile_tolerance
        lower_bound, upper_bound = low - tolerance, high + tolerance

        breaches = int(np.sum((forecast < lower_bound) | (forecast > upper_bound)))
        if breaches:
            return self._failed(
                RejectionReason.UNREALISTIC_FORECAST_SPIKES,
                f"{breaches} forecast value(s) fall outside the historical "
                f"[{lower_bound:.2f}, {upper_bound:.2f}] band.",
                breaches=breaches,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        return self._passed(breaches=0, lower_bound=lower_bound, upper_bound=upper_bound)


class IqrBoundsRule(ValidationRule):
    """Forecast escapes an IQR-based bound on history.

    A robust alternative to the percentile band: with a heavy-tailed
    history, quartiles describe the bulk of the distribution better than
    extreme percentiles do.
    """

    rule_id = "iqr_bounds"
    rule_name = "Historical Bound Violation"

    # Store overfitting rule thresholds
    def __init__(self, config: OverfittingRulesConfig) -> None:
        self._config = config

    # Whether IQR bounds checking is enabled
    def is_enabled(self) -> bool:
        return self._config.iqr_bounds_enabled

    # Reject values that escape the IQR-based bound
    def evaluate(self, history: np.ndarray, forecast: np.ndarray) -> RuleOutcome:
        q1 = float(np.percentile(history, 25))
        q3 = float(np.percentile(history, 75))
        iqr = q3 - q1

        if iqr == 0:
            return self._passed(iqr=0.0)

        lower_bound = q1 - self._config.iqr_multiplier * iqr
        upper_bound = q3 + self._config.iqr_multiplier * iqr

        breaches = int(np.sum((forecast < lower_bound) | (forecast > upper_bound)))
        if breaches:
            return self._failed(
                RejectionReason.HISTORICAL_BOUND_VIOLATION,
                f"{breaches} forecast value(s) fall outside the IQR-based "
                f"[{lower_bound:.2f}, {upper_bound:.2f}] bound.",
                breaches=breaches,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        return self._passed(breaches=0, lower_bound=lower_bound, upper_bound=upper_bound)


class AbnormalGrowthRule(ValidationRule):
    """Forecast level departs sharply from the recent historical level."""

    rule_id = "abnormal_growth"
    rule_name = "Abnormal Growth Rate"

    # Store overfitting rule thresholds
    def __init__(self, config: OverfittingRulesConfig) -> None:
        self._config = config

    # Whether growth-rate checking is enabled
    def is_enabled(self) -> bool:
        return self._config.growth_rate_enabled

    # Reject a forecast mean far from the recent historical mean
    def evaluate(self, history: np.ndarray, forecast: np.ndarray) -> RuleOutcome:
        # Compare against recent history rather than the whole series, so a
        # long-past regime does not excuse an implausible jump today.
        reference = history[-self._config.growth_reference_periods :]
        reference_mean = float(np.mean(reference))

        if reference_mean == 0:
            return self._passed(reference_mean=0.0)

        forecast_mean = float(np.mean(forecast))
        ratio = abs(forecast_mean / reference_mean)

        upper = self._config.max_growth_ratio
        lower = 1.0 / upper if upper else 0.0

        if ratio > upper or ratio < lower:
            return self._failed(
                RejectionReason.ABNORMAL_GROWTH,
                f"Forecast mean is {ratio:.2f}x the recent historical mean, outside the "
                f"[{lower:.2f}, {upper:.2f}] range.",
                growth_ratio=ratio,
                lower_limit=lower,
                upper_limit=upper,
            )
        return self._passed(growth_ratio=ratio)


class ForecastVarianceRule(ValidationRule):
    """Dispersion across the horizon is inconsistent with history."""

    rule_id = "forecast_variance"
    rule_name = "High Forecast Variance"

    # Store overfitting rule thresholds
    def __init__(self, config: OverfittingRulesConfig) -> None:
        self._config = config

    # Whether forecast-variance checking is enabled
    def is_enabled(self) -> bool:
        return self._config.forecast_variance_enabled

    # Reject dispersion inconsistent with history
    def evaluate(self, history: np.ndarray, forecast: np.ndarray) -> RuleOutcome:
        # Coefficient of variation compares dispersion relative to level, so
        # the check is unaffected by the units of the target.
        historical_cv = _coefficient_of_variation(history)
        forecast_cv = _coefficient_of_variation(forecast)

        if historical_cv is None or forecast_cv is None or historical_cv == 0:
            return self._passed(historical_cv=historical_cv, forecast_cv=forecast_cv)

        ratio = forecast_cv / historical_cv
        if ratio > self._config.max_variance_ratio:
            return self._failed(
                RejectionReason.HIGH_FORECAST_VARIANCE,
                f"Forecast dispersion is {ratio:.2f}x historical dispersion, above the "
                f"{self._config.max_variance_ratio}x limit.",
                variance_ratio=ratio,
                threshold=self._config.max_variance_ratio,
            )
        return self._passed(variance_ratio=ratio)


# ----------------------------------------------------------------------
# Underfitting signals (Section 6.5.2)
# ----------------------------------------------------------------------


class FlatForecastRule(ValidationRule):
    """Forecast is constant, or nearly so, against a varying history."""

    rule_id = "flat_forecast"
    rule_name = "Flat Forecast"

    # Store underfitting rule thresholds
    def __init__(self, config: UnderfittingRulesConfig) -> None:
        self._config = config

    # Whether flat-forecast checking is enabled
    def is_enabled(self) -> bool:
        return self._config.flat_forecast_enabled

    # Reject a forecast that stays flat against a varying history
    def evaluate(self, history: np.ndarray, forecast: np.ndarray) -> RuleOutcome:
        historical_std = float(np.std(history))
        forecast_std = float(np.std(forecast))

        # A genuinely flat history makes a flat forecast correct.
        if historical_std == 0:
            return self._passed(historical_std=0.0, forecast_std=forecast_std)

        ratio = forecast_std / historical_std
        if ratio < self._config.min_volatility_ratio:
            return self._failed(
                RejectionReason.FLAT_FORECAST,
                f"Forecast variation is only {ratio:.3f}x historical variation, below the "
                f"{self._config.min_volatility_ratio} minimum.",
                volatility_ratio=ratio,
                threshold=self._config.min_volatility_ratio,
            )
        return self._passed(volatility_ratio=ratio)


class ExcessiveSmoothingRule(ValidationRule):
    """Forecast varies, but far less than the series it models.

    Distinct from the flat check: this catches a forecast that retains some
    shape while compressing the amplitude of the series away.
    """

    rule_id = "excessive_smoothing"
    rule_name = "Excessive Smoothing"

    # Store underfitting rule thresholds
    def __init__(self, config: UnderfittingRulesConfig) -> None:
        self._config = config

    # Whether excessive-smoothing checking is enabled
    def is_enabled(self) -> bool:
        return self._config.excessive_smoothing_enabled

    # Reject a forecast that compresses historical amplitude away
    def evaluate(self, history: np.ndarray, forecast: np.ndarray) -> RuleOutcome:
        historical_std = float(np.std(history))
        forecast_std = float(np.std(forecast))

        if historical_std == 0:
            return self._passed(historical_std=0.0)

        ratio = forecast_std / historical_std
        if ratio < self._config.min_smoothing_ratio:
            return self._failed(
                RejectionReason.EXCESSIVE_SMOOTHING,
                f"Forecast retains only {ratio:.3f} of historical variance, below the "
                f"{self._config.min_smoothing_ratio} minimum.",
                smoothing_ratio=ratio,
                threshold=self._config.min_smoothing_ratio,
            )
        return self._passed(smoothing_ratio=ratio)


class MissingSeasonalityRule(ValidationRule):
    """Seasonality present in history has vanished from the forecast.

    Only applied when the history actually is seasonal, measured by
    autocorrelation at the seasonal lag — demanding seasonality of a
    non-seasonal series would reject correct forecasts.
    """

    rule_id = "missing_seasonality"
    rule_name = "Missing Seasonality"

    # Store underfitting rule thresholds
    def __init__(self, config: UnderfittingRulesConfig) -> None:
        self._config = config

    # Whether seasonality checking is enabled
    def is_enabled(self) -> bool:
        return self._config.seasonality_enabled

    # Reject a forecast that drops seasonality present in history
    def evaluate(self, history: np.ndarray, forecast: np.ndarray) -> RuleOutcome:
        period = self._config.seasonal_period

        # Two full cycles are the minimum needed to speak of seasonality.
        if len(history) < period * 2:
            return self._passed(seasonality_strength=None, reason_skipped="insufficient history")

        strength = _autocorrelation(history, period)
        if strength is None or strength < self._config.seasonality_strength_threshold:
            return self._passed(seasonality_strength=strength)

        # History is seasonal: require the forecast to show a comparable
        # swing to the one history exhibits over a cycle.
        historical_range = float(np.max(history[-period:]) - np.min(history[-period:]))
        forecast_range = float(np.max(forecast) - np.min(forecast))

        if historical_range == 0:
            return self._passed(historical_range=0.0)

        ratio = forecast_range / historical_range
        if ratio < self._config.min_seasonal_range_ratio:
            return self._failed(
                RejectionReason.MISSING_SEASONALITY,
                f"History is seasonal (autocorrelation {strength:.2f} at lag {period}) but the "
                f"forecast spans only {ratio:.2f} of the historical seasonal range.",
                seasonality_strength=strength,
                seasonal_range_ratio=ratio,
                threshold=self._config.min_seasonal_range_ratio,
            )
        return self._passed(seasonality_strength=strength, seasonal_range_ratio=ratio)


class MissingTrendRule(ValidationRule):
    """A clear historical trend is absent from the forecast."""

    rule_id = "missing_trend"
    rule_name = "Missing Trend"

    # Store underfitting rule thresholds
    def __init__(self, config: UnderfittingRulesConfig) -> None:
        self._config = config

    # Whether trend checking is enabled
    def is_enabled(self) -> bool:
        return self._config.trend_enabled

    # Reject a forecast that drops a clear historical trend
    def evaluate(self, history: np.ndarray, forecast: np.ndarray) -> RuleOutcome:
        historical_slope = _normalized_slope(history)
        if historical_slope is None:
            return self._passed(historical_slope=None)

        # Only a materially trending history imposes an obligation.
        if abs(historical_slope) < self._config.min_historical_trend_strength:
            return self._passed(historical_slope=historical_slope)

        forecast_slope = _normalized_slope(forecast)
        if forecast_slope is None:
            return self._passed(historical_slope=historical_slope, forecast_slope=None)

        retention = abs(forecast_slope) / abs(historical_slope)
        if retention < self._config.min_trend_retention_ratio:
            return self._failed(
                RejectionReason.MISSING_TREND,
                f"History trends at {historical_slope:.3f} per period but the forecast retains "
                f"only {retention:.2f} of that slope.",
                historical_slope=historical_slope,
                forecast_slope=forecast_slope,
                trend_retention=retention,
            )
        return self._passed(
            historical_slope=historical_slope,
            forecast_slope=forecast_slope,
            trend_retention=retention,
        )


# ----------------------------------------------------------------------
# Shared statistics
# ----------------------------------------------------------------------


# Standard deviation relative to the mean, or None when undefined
def _coefficient_of_variation(values: np.ndarray) -> float | None:
    mean = float(np.mean(values))
    if mean == 0:
        return None
    return float(np.std(values) / abs(mean))


# Autocorrelation of `values` at `lag`, or None when unmeasurable
def _autocorrelation(values: np.ndarray, lag: int) -> float | None:
    if len(values) <= lag:
        return None

    centred = values - np.mean(values)
    denominator = float(np.sum(centred**2))
    if denominator == 0:
        return None

    numerator = float(np.sum(centred[lag:] * centred[:-lag]))
    return numerator / denominator


# Least-squares slope expressed relative to the series level
def _normalized_slope(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None

    mean = float(np.mean(values))
    if mean == 0:
        return None

    positions = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(positions, values, 1)[0])
    return slope / abs(mean)
