"""Final Production Model Selection result models — the output contract of
Section 6.9, and of Phase 7B as a whole. This is what Explainable AI and
MLflow Logging consume next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from forecast_engine.s09_drift.drift_report import DriftValidationResult
from forecast_engine.s06_evaluation.evaluation_report import ForwardForecast


class FinalSelectionStatus(str, Enum):
    """How selection ended for one group.

    Both FALLBACK_USED and NO_MODEL_AVAILABLE mean no ranked candidate
    survived drift validation, but only the first produced a usable forecast;
    the second means even the fallback could not be trained.
    """

    SELECTED = "Selected"
    FALLBACK_USED = "Fallback Model Used"
    NO_MODEL_AVAILABLE = "No Model Available"
    FAILED = "Failed"


@dataclass
class RejectedCandidate:
    """One ranked candidate that did not become the production model."""

    model_name: str
    reason: str

    # Serialize to a plain dict
    def to_dict(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "reason": self.reason}


@dataclass
class ProductionModelResult:
    """The final production decision for one forecasting group."""

    group_id: str
    status: FinalSelectionStatus
    model_name: str | None = None
    key_values: dict[str, Any] = field(default_factory=dict)

    composite_score: float | None = None
    original_backtest_rank: int | None = None
    final_rank: int | None = None

    drift_validation: DriftValidationResult | None = None
    fallback_used: bool = False
    rejected_candidates: list[RejectedCandidate] = field(default_factory=list)
    error: str | None = None

    # Fallback audit trail (Section 6.9). Populated only when the fallback
    # path ran: what triggered it, and which ranked candidates were
    # considered before it did. Kept alongside `rejected_candidates`
    # (per-candidate detail) so a reader gets both the one-line trigger and
    # the full evidence without recomputing either.
    fallback_model: str | None = None
    fallback_trigger: str | None = None
    original_candidates: list[str] = field(default_factory=list)

    # The final production model's own 12-month forward forecast — carried
    # here (rather than requiring a second lookup into Phase 7A's
    # EvaluationReport, which has no entry for a fallback model) so this
    # result is a complete, self-contained record of the production
    # decision. Populated for both a ranked winner and a fallback.
    forecast: ForwardForecast | None = None

    # The fitted estimator behind this decision, when the fallback path
    # produced it. A ranked winner's estimator already lives on its
    # training record; the fallback's is fitted here during selection and
    # would otherwise be discarded, so persistence would have to refit it.
    # Excluded from `to_dict()` — a live in-process object, not part of
    # the run's serializable record — exactly as `TrainedModel` treats its
    # own. Holding it changes no decision this stage makes.
    fitted_model: Any = None

    # Serialize the full production decision to a plain dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_group": self.group_id,
            "key_values": self.key_values,
            "final_production_model": self.model_name,
            "composite_ranking_score": _round(self.composite_score),
            "original_backtesting_rank": self.original_backtest_rank,
            "final_rank": self.final_rank,
            "selected_drift_algorithm": (
                self.drift_validation.algorithm_selection.algorithm.value if self.drift_validation else None
            ),
            "drift_statistic": _round(self.drift_validation.drift_statistic) if self.drift_validation else None,
            "dynamic_threshold_method": (
                self.drift_validation.threshold.method.value if self.drift_validation else None
            ),
            "dynamic_threshold_value": _round(self.drift_validation.threshold.value) if self.drift_validation else None,
            "drift_validation_result": self.drift_validation.to_dict() if self.drift_validation else None,
            "final_selection_status": self.status.value,
            "fallback_flag": self.fallback_used,
            "fallback_model": self.fallback_model,
            "fallback_trigger": self.fallback_trigger,
            "original_candidates": list(self.original_candidates),
            "failure_reasons": [candidate.to_dict() for candidate in self.rejected_candidates],
            "rejected_candidates": [candidate.to_dict() for candidate in self.rejected_candidates],
            "forecast": self.forecast.to_dict() if self.forecast else None,
            "error": self.error,
        }


@dataclass
class ProductionSelectionReport:
    """Aggregate outcome of Final Production Model Selection."""

    results: list[ProductionModelResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    # Count groups whose ranked candidate was selected
    @property
    def selected_count(self) -> int:
        return sum(1 for result in self.results if result.status is FinalSelectionStatus.SELECTED)

    # Count groups that fell back to the configured fallback model
    @property
    def fallback_count(self) -> int:
        return sum(1 for result in self.results if result.status is FinalSelectionStatus.FALLBACK_USED)

    # Count groups with no usable model at all
    @property
    def unavailable_count(self) -> int:
        return sum(1 for result in self.results if result.status is FinalSelectionStatus.NO_MODEL_AVAILABLE)

    # Count groups whose selection raised an error
    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.results if result.status is FinalSelectionStatus.FAILED)

    # Serialize the aggregate report to a plain dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": len(self.results),
            "selected": self.selected_count,
            "fallback_used": self.fallback_count,
            "no_model_available": self.unavailable_count,
            "failed": self.failed_count,
            "duration_seconds": round(self.duration_seconds, 3),
            "results": [result.to_dict() for result in self.results],
        }


# Round a value, preserving None
def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)
