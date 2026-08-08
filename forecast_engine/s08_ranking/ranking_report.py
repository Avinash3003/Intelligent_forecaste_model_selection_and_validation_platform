"""Model Ranking result models — the output contract of Section 6.6.

One `RankedModel` per surviving (forecast group, model) pair, carrying the
composite score plus every component that fed it, so the score is never a
black box. `RankingReport.top_ranked_by_group()` is Section 6.7's direct
input — the model each group's Drift Validation is run against first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BacktestScoreBreakdown:
    """Normalized, weighted view of one model's backtesting metrics.

    `normalized` values are in [0, 1], min-max scaled *within the group's
    candidate set* so a metric's native units never leak into the score —
    1.0 always means "best of this group's survivors" on that metric.
    """

    normalized: dict[str, float] = field(default_factory=dict)
    score: float = 0.0

    # Serialize to a plain dict
    def to_dict(self) -> dict[str, Any]:
        return {"normalized_metrics": {k: round(v, 4) for k, v in self.normalized.items()}, "score": round(self.score, 4)}


@dataclass
class StabilityScoreBreakdown:
    """Normalized, weighted view of one model's forecast stability."""

    variance_score: float = 0.0
    smoothness_score: float = 0.0
    interval_score: float = 0.0
    score: float = 0.0

    # Serialize to a plain dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "variance_score": round(self.variance_score, 4),
            "smoothness_score": round(self.smoothness_score, 4),
            "interval_score": round(self.interval_score, 4),
            "score": round(self.score, 4),
        }


@dataclass
class ShapScoreBreakdown:
    """SHAP (or native/permutation-importance) consistency signal.

    `method` records which path produced this score — `shap`, `permutation`
    or `univariate_fallback` — so the ranking output is transparent about
    how a model that does not support SHAP directly was still evaluated.
    """

    method: str = "univariate_fallback"
    importances: dict[str, float] = field(default_factory=dict)
    stability: float | None = None
    plausibility: float | None = None
    window_consistency: float | None = None
    score: float = 0.0
    skipped_reason: str | None = None

    # Serialize to a plain dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "importances": {k: round(v, 4) for k, v in self.importances.items()},
            "stability": _round(self.stability),
            "plausibility": _round(self.plausibility),
            "window_consistency": _round(self.window_consistency),
            "score": round(self.score, 4),
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class RankedModel:
    """One surviving model's complete ranking record for its forecast group."""

    group_id: str
    model_name: str
    key_values: dict[str, Any] = field(default_factory=dict)

    backtest: BacktestScoreBreakdown = field(default_factory=BacktestScoreBreakdown)
    stability: StabilityScoreBreakdown = field(default_factory=StabilityScoreBreakdown)
    shap: ShapScoreBreakdown = field(default_factory=ShapScoreBreakdown)

    composite_score: float = 0.0
    original_backtest_rank: int = 0
    final_composite_rank: int = 0

    # Serialize to a plain dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "model_name": self.model_name,
            "key_values": self.key_values,
            "backtest": self.backtest.to_dict(),
            "stability": self.stability.to_dict(),
            "shap": self.shap.to_dict(),
            "composite_score": round(self.composite_score, 4),
            "original_backtest_rank": self.original_backtest_rank,
            "final_composite_rank": self.final_composite_rank,
        }


@dataclass
class RankingReport:
    """Aggregate outcome of ranking every group's surviving models."""

    rankings: dict[str, list[RankedModel]] = field(default_factory=dict)
    duration_seconds: float = 0.0

    # Candidates for one group, ordered by final composite rank
    def for_group(self, group_id: str) -> list[RankedModel]:
        return self.rankings.get(group_id, [])

    # The #1 ranked candidate per group (Drift Validation's first candidate)
    def top_ranked_by_group(self) -> dict[str, "RankedModel | None"]:
        return {group_id: (models[0] if models else None) for group_id, models in self.rankings.items()}

    # Serialize to a plain dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "groups_ranked": len(self.rankings),
            "duration_seconds": round(self.duration_seconds, 3),
            "rankings": {
                group_id: [model.to_dict() for model in models] for group_id, models in self.rankings.items()
            },
        }


# Round a nullable float for serialization
def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)
