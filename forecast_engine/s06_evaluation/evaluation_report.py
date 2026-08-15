"""What evaluation produces for each (group, model) pair.

Backtest metrics, the forward forecast, whether it survived validation and,
if not, the structured reasons why.

Rejection reasons are codes, never prose — the dashboard and the
explainability layer both build on them. Every to_dict() yields plain JSON,
so a result can be logged or returned without conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from forecast_engine.config.evaluation_config import RejectionReason
from forecast_engine.s06_evaluation.metrics import ForecastMetrics


class EvaluationStatus(str, Enum):
    """How evaluating one model on one group ended.

    ELIMINATED is a judgement (the forecast was implausible); FAILED is a
    malfunction (it could not backtest or forecast at all). Ranking and the
    dashboard treat them differently.
    """

    SURVIVED = "Survived"
    ELIMINATED = "Eliminated"
    FAILED = "Failed"
    SKIPPED = "Skipped"


@dataclass
class BacktestWindowResult:
    """Metrics from one backtest fold."""

    window_index: int
    train_start: str | None
    train_end: str | None
    test_start: str | None
    test_end: str | None
    train_size: int
    metrics: ForecastMetrics

    # Serialize this backtest window's result to a dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "window_index": self.window_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_size": self.train_size,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class BacktestResult:
    """One pair's backtest, in three views: overall, per fold, and per horizon
    step — the last showing how accuracy decays with distance."""

    strategy: str
    windows: list[BacktestWindowResult] = field(default_factory=list)
    overall: ForecastMetrics = field(default_factory=ForecastMetrics)
    per_horizon: dict[int, ForecastMetrics] = field(default_factory=dict)
    skipped_reason: str | None = None

    # Number of folds actually evaluated
    @property
    def window_count(self) -> int:
        return len(self.windows)

    # Serialize the backtest result to a dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "window_count": self.window_count,
            "overall": self.overall.to_dict(),
            "per_horizon": {str(step): metrics.to_dict() for step, metrics in self.per_horizon.items()},
            "windows": [window.to_dict() for window in self.windows],
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class ForwardForecast:
    """A model's forward forecast over the configured horizon."""

    dates: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    lower: list[float] | None = None
    upper: list[float] | None = None

    # Number of periods this forecast covers
    @property
    def horizon(self) -> int:
        return len(self.values)

    # Whether confidence intervals are present
    @property
    def has_intervals(self) -> bool:
        return self.lower is not None and self.upper is not None

    # Serialize the forward forecast to a dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "dates": list(self.dates),
            "values": [float(value) for value in self.values],
            "lower": [float(value) for value in self.lower] if self.lower is not None else None,
            "upper": [float(value) for value in self.upper] if self.upper is not None else None,
        }


@dataclass
class RuleOutcome:
    """Result of one independent elimination rule (Section 6.5.3)."""

    rule_id: str
    rule_name: str
    passed: bool
    reason: RejectionReason | None = None
    detail: str | None = None
    measurements: dict[str, Any] = field(default_factory=dict)

    # Serialize the rule outcome to a dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "passed": self.passed,
            "reason": self.reason.value if self.reason else None,
            "detail": self.detail,
            "measurements": {
                key: (round(value, 4) if isinstance(value, float) else value)
                for key, value in self.measurements.items()
            },
        }


@dataclass
class ForwardValidationResult:
    """Verdict of the elimination framework for one forecast."""

    passed: bool = True
    rule_outcomes: list[RuleOutcome] = field(default_factory=list)
    skipped_reason: str | None = None

    # Rule outcomes that failed
    @property
    def failed_rules(self) -> list[RuleOutcome]:
        return [outcome for outcome in self.rule_outcomes if not outcome.passed]

    # Every distinct reason this forecast was rejected, in rule order
    @property
    def rejection_reasons(self) -> list[RejectionReason]:
        seen: list[RejectionReason] = []
        for outcome in self.failed_rules:
            if outcome.reason and outcome.reason not in seen:
                seen.append(outcome.reason)
        return seen

    # Serialize the forward validation result to a dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "rejection_reasons": [reason.value for reason in self.rejection_reasons],
            "rule_outcomes": [outcome.to_dict() for outcome in self.rule_outcomes],
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class EvaluationResult:
    """Everything Phase 7A produces for one group/model pair."""

    group_id: str
    model_name: str
    status: EvaluationStatus
    key_values: dict[str, Any] = field(default_factory=dict)
    backtest: BacktestResult | None = None
    forecast: ForwardForecast | None = None
    validation: ForwardValidationResult | None = None
    error: str | None = None

    # Whether this pair survived to ranking
    @property
    def survived(self) -> bool:
        return self.status is EvaluationStatus.SURVIVED

    # Rejection reason codes, or empty when not validated
    @property
    def rejection_reasons(self) -> list[str]:
        if not self.validation:
            return []
        return [reason.value for reason in self.validation.rejection_reasons]

    # Serialize the evaluation result to a dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "model_name": self.model_name,
            "status": self.status.value,
            "key_values": self.key_values,
            "backtest": self.backtest.to_dict() if self.backtest else None,
            "forecast": self.forecast.to_dict() if self.forecast else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "rejection_reasons": self.rejection_reasons,
            "error": self.error,
        }


@dataclass
class EvaluationReport:
    """The evaluation stage's overall outcome.

    The *_seconds fields break the total down, because Evaluate and Train
    look like comparable stages in the summary while evaluation does several
    fits per pair where training does one.
    """

    results: list[EvaluationResult] = field(default_factory=list)
    groups_evaluated: int = 0
    duration_seconds: float = 0.0

    # Wall-clock time spent inside each of the three sub-phases every
    # (group, model) pair goes through, summed across all pairs.
    backtest_seconds: float = 0.0
    forecast_generation_seconds: float = 0.0
    validation_seconds: float = 0.0

    # Total model fits performed across every backtest fold plus every
    # forward-forecast generation. Rolling/expanding backtesting requires a
    # fresh fit per fold (Section 6.4 — reusing a fold's fit would leak
    # future data into its own score), so this is genuinely proportional to
    # `backtest_windows_evaluated`, not a bug by itself; it is what makes
    # Evaluation's cost explicit rather than implicit in a single duration.
    model_fit_count: int = 0

    # Backtest folds actually evaluated, summed across every pair — the
    # single largest driver of `backtest_seconds`.
    backtest_windows_evaluated: int = 0

    # Forward forecasts produced by reusing the model Training already
    # fit on this exact series, instead of fitting an identical model a
    # second time. Section 6.4's backtest folds are never reused (each
    # trains on a different, smaller window — reusing them would be
    # invalid), but the final full-series forecast fit is otherwise a
    # verbatim duplicate of Training's own fit: same series, same tuned
    # parameters. See `ForwardForecastGenerator`.
    forecasts_reused: int = 0
    forecasts_refit: int = 0

    # Count of results with SURVIVED status
    @property
    def survived_count(self) -> int:
        return sum(1 for result in self.results if result.status is EvaluationStatus.SURVIVED)

    # Count of results with ELIMINATED status
    @property
    def eliminated_count(self) -> int:
        return sum(1 for result in self.results if result.status is EvaluationStatus.ELIMINATED)

    # Count of results with FAILED status
    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.results if result.status is EvaluationStatus.FAILED)

    # Count of results with SKIPPED status
    @property
    def skipped_count(self) -> int:
        return sum(1 for result in self.results if result.status is EvaluationStatus.SKIPPED)

    # Models eligible for ranking — Phase 7B's direct input
    def surviving_models(self) -> list[EvaluationResult]:
        return [result for result in self.results if result.status is EvaluationStatus.SURVIVED]

    # Survivors indexed by forecasting group
    def surviving_by_group(self) -> dict[str, list[EvaluationResult]]:
        grouped: dict[str, list[EvaluationResult]] = {}
        for result in self.surviving_models():
            grouped.setdefault(result.group_id, []).append(result)
        return grouped

    # Serialize the aggregate evaluation report to a dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "groups_evaluated": self.groups_evaluated,
            "total_results": len(self.results),
            "survived": self.survived_count,
            "eliminated": self.eliminated_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "duration_seconds": round(self.duration_seconds, 3),
            "timing_breakdown": {
                "backtest_seconds": round(self.backtest_seconds, 3),
                "forecast_generation_seconds": round(self.forecast_generation_seconds, 3),
                "validation_seconds": round(self.validation_seconds, 3),
                "model_fit_count": self.model_fit_count,
                "backtest_windows_evaluated": self.backtest_windows_evaluated,
                "forecasts_reused": self.forecasts_reused,
                "forecasts_refit": self.forecasts_refit,
            },
            "results": [result.to_dict() for result in self.results],
        }
