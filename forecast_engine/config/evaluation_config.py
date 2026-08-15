"""Backtesting, metrics and forward-validation settings.

Nothing here is an absolute magnitude: every elimination rule is a ratio or
percentile measured against the series being judged, which is what lets the
same rules apply to unit sales and to revenue without retuning.

Each rule carries its own enabled flag, so one can be switched off without
disturbing the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BacktestStrategy(str, Enum):
    """How training windows advance.

    ROLLING slides a fixed-length window forward; EXPANDING grows from a
    fixed origin, using all history available at each point.
    """

    ROLLING = "rolling"
    EXPANDING = "expanding"


class RejectionReason(str, Enum):
    """Why a model was eliminated, as codes rather than prose.

    The dashboard and the explainability layer both key off these, so the
    reason stays machine-readable.
    """

    EXCESSIVE_FORECAST_VOLATILITY = "Excessive Forecast Volatility"
    UNREALISTIC_FORECAST_SPIKES = "Unrealistic Forecast Spikes"
    ABNORMAL_GROWTH = "Abnormal Growth Rate"
    HIGH_FORECAST_VARIANCE = "High Forecast Variance"
    FLAT_FORECAST = "Flat Or Near-Constant Forecast"
    MISSING_SEASONALITY = "Missing Seasonality"
    MISSING_TREND = "Missing Trend"
    EXCESSIVE_SMOOTHING = "Excessive Smoothing"
    HISTORICAL_BOUND_VIOLATION = "Historical Bound Violation"
    NON_FINITE_FORECAST = "Non-Finite Forecast Values"


@dataclass(frozen=True)
class BacktestConfig:
    """Controls the rolling/expanding backtest harness."""

    strategy: BacktestStrategy = BacktestStrategy.ROLLING

    # Periods predicted in each fold. Kept separate from the *forward*
    # forecast horizon: a 12-month backtest fold would consume a third of a
    # 36-month series, so short histories can be evaluated on a shorter fold
    # while still producing a full 12-month forward forecast.
    horizon: int = 3

    # Observations required before the first fold can be scored.
    min_train_size: int = 12

    # How far the window advances between folds. Defaults to the fold
    # horizon, giving non-overlapping test windows.
    step: int | None = None

    # Upper bound on folds, so a long series does not spend unbounded time
    # in backtesting.
    max_windows: int = 5

    # Fixed training length for ROLLING. None means "use min_train_size".
    rolling_window_size: int | None = None


@dataclass(frozen=True)
class MetricsConfig:
    """Controls metric computation (Section 6.4)."""

    # Percentage metrics divide by the actual value, which is undefined at
    # zero. Those observations are excluded from MAPE rather than allowed to
    # produce an infinite score; WMAPE (a ratio of sums) is unaffected and
    # is why the platform prefers it for accuracy reporting (Section 10).
    exclude_zero_actuals_from_mape: bool = True

    # Metrics are also reported per horizon step, as Section 6.4 requires.
    compute_per_horizon: bool = True


@dataclass(frozen=True)
class OverfittingRulesConfig:
    """Thresholds for the overfitting signals in Section 6.5.1."""

    # Forecast standard deviation may not exceed this multiple of the
    # historical standard deviation.
    volatility_ratio_enabled: bool = True
    max_volatility_ratio: float = 2.5

    # Forecast values must stay inside the historical percentile band,
    # widened by a tolerance so a genuine continuation of trend is not
    # mistaken for a spike.
    spike_detection_enabled: bool = True
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    percentile_tolerance: float = 0.25

    # IQR-based bound, an alternative view of the same idea that is robust
    # to a heavy-tailed history.
    iqr_bounds_enabled: bool = True
    iqr_multiplier: float = 3.0

    # Mean forecast level may not depart from the recent historical level by
    # more than this share.
    growth_rate_enabled: bool = True
    max_growth_ratio: float = 1.5
    growth_reference_periods: int = 12

    # Coefficient of variation across the horizon, relative to history.
    forecast_variance_enabled: bool = True
    max_variance_ratio: float = 3.0


@dataclass(frozen=True)
class UnderfittingRulesConfig:
    """Thresholds for the underfitting signals in Section 6.5.2."""

    # A forecast whose variation collapses relative to history has stopped
    # modelling the series.
    flat_forecast_enabled: bool = True
    min_volatility_ratio: float = 0.05

    # Distinct from the flat check: variance present but heavily compressed.
    excessive_smoothing_enabled: bool = True
    min_smoothing_ratio: float = 0.15

    # Seasonality is only required of a forecast when history actually shows
    # it, measured by autocorrelation at the seasonal lag.
    seasonality_enabled: bool = True
    seasonal_period: int = 12
    seasonality_strength_threshold: float = 0.3
    min_seasonal_range_ratio: float = 0.25

    # Likewise trend: only required when history has a meaningful one.
    trend_enabled: bool = True
    min_historical_trend_strength: float = 0.05
    min_trend_retention_ratio: float = 0.1


@dataclass(frozen=True)
class ForwardValidationConfig:
    """Root configuration for forward-forecast validation (Section 6.5)."""

    enabled: bool = True
    overfitting: OverfittingRulesConfig = field(default_factory=OverfittingRulesConfig)
    underfitting: UnderfittingRulesConfig = field(default_factory=UnderfittingRulesConfig)

    # Minimum history before validation is meaningful. Below this the rules
    # would be comparing a forecast against too little evidence, so the
    # model is passed through rather than judged on noise.
    min_history_for_validation: int = 12


@dataclass(frozen=True)
class EvaluationConfig:
    """Root evaluation configuration carried through the phase."""

    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    forward_validation: ForwardValidationConfig = field(default_factory=ForwardValidationConfig)

    # Periods produced by the forward forecast (Section 6.5 / 3: minimum
    # twelve-month horizon).
    forecast_horizon: int = 12

    # Return the standard evaluation configuration
    @classmethod
    def default(cls) -> "EvaluationConfig":
        return cls()

    # Build from a nested mapping, ignoring unknown keys
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationConfig":
        forward = payload.get("forward_validation") or {}
        return cls(
            backtest=_build_block(BacktestConfig, payload.get("backtest")),
            metrics=_build_block(MetricsConfig, payload.get("metrics")),
            forward_validation=ForwardValidationConfig(
                enabled=forward.get("enabled", True),
                overfitting=_build_block(OverfittingRulesConfig, forward.get("overfitting")),
                underfitting=_build_block(UnderfittingRulesConfig, forward.get("underfitting")),
                min_history_for_validation=forward.get("min_history_for_validation", 12),
            ),
            forecast_horizon=payload.get("forecast_horizon", 12),
        )


# Instantiate one config block, keeping only keys it declares
def _build_block(block_type: type, values: dict[str, Any] | None) -> Any:
    if not values:
        return block_type()

    allowed = {f.name for f in block_type.__dataclass_fields__.values()}
    return block_type(**{key: value for key, value in values.items() if key in allowed})
