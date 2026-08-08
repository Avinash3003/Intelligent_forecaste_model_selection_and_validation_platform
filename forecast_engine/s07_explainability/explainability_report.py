"""Explainability result models — the output contract of Section 6.10.

One `ExplainabilityResult` per (forecast group, model) pair, generated once
and reused by both Model Ranking and the LLM Insight Engine, so the two
consumers never compute — or disagree about — the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forecast_engine.s08_ranking.ranking_report import ShapScoreBreakdown


@dataclass
class StabilityMetrics:
    """Structured explainability metadata (Section 6.10, "SHAP Stability").

    Every field is a measurement or a code, never prose — narrative
    interpretation of these numbers is the LLM Insight Engine's job
    (Section 6.12), not this stage's.
    """

    window_stability: float | None = None
    horizon_stability: float | None = None
    plausibility: float | None = None
    dominant_feature: str | None = None
    dominant_feature_consistency: float | None = None
    significant_changes: list[str] = field(default_factory=list)
    composite_score: float = 0.0

    # Serialize stability metrics to a rounded dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "window_stability": _round(self.window_stability),
            "horizon_stability": _round(self.horizon_stability),
            "plausibility": _round(self.plausibility),
            "dominant_feature": self.dominant_feature,
            "dominant_feature_consistency": _round(self.dominant_feature_consistency),
            "significant_changes": list(self.significant_changes),
            "composite_score": round(self.composite_score, 4),
        }


@dataclass
class ExplainabilityResult:
    """Everything Section 6.10 produces for one (group, model) pair."""

    group_id: str
    model_name: str
    key_values: dict[str, Any] = field(default_factory=dict)

    method: str = "univariate_fallback"
    global_importance: dict[str, float] = field(default_factory=dict)
    local_importance: list[dict[str, Any]] = field(default_factory=list)
    shap_values_available: bool = False
    stability: StabilityMetrics = field(default_factory=StabilityMetrics)
    skipped_reason: str | None = None

    # Alias for `global_importance` — exposed under both output names
    @property
    def feature_importance(self) -> dict[str, float]:
        return self.global_importance

    # The compact view Model Ranking's composite score consumes
    def ranking_summary(self) -> ShapScoreBreakdown:
        return ShapScoreBreakdown(
            method=self.method,
            importances=self.global_importance,
            stability=self.stability.window_stability,
            plausibility=self.stability.plausibility,
            window_consistency=self.stability.window_stability,
            score=self.stability.composite_score,
            skipped_reason=self.skipped_reason,
        )

    # Serialize the explainability result to a dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "model_name": self.model_name,
            "key_values": self.key_values,
            "method": self.method,
            "global_importance": {k: round(v, 4) for k, v in self.global_importance.items()},
            "feature_importance": {k: round(v, 4) for k, v in self.feature_importance.items()},
            "local_importance": self.local_importance,
            "shap_values_available": self.shap_values_available,
            "stability_metrics": self.stability.to_dict(),
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class ExplainabilityReport:
    """Aggregate outcome of explainability generation for one run."""

    results: list[ExplainabilityResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    # Look up one (group, model) pair's result
    def for_pair(self, group_id: str, model_name: str) -> ExplainabilityResult | None:
        for result in self.results:
            if result.group_id == group_id and result.model_name == model_name:
                return result
        return None

    # Serialize the aggregate explainability report to a dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_results": len(self.results),
            "duration_seconds": round(self.duration_seconds, 3),
            "results": [result.to_dict() for result in self.results],
        }


# Round a metric value, passing through None
def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)
