"""Thresholds for the SHAP / feature-importance engine.

Explainability runs as its own stage ahead of ranking, so its settings live
here. RankingConfig keeps only the weight deciding how much the signal
counts, not how it is produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExplainabilityConfig:
    """Governs both the SHAP path (tree models, when shap is installed) and the
    permutation/univariate fallbacks, so no surviving model is skipped."""

    enabled: bool = True

    # Resamples used to measure how much an importance ranking moves when
    # the training data is perturbed slightly ("SHAP stability").
    bootstrap_iterations: int = 3

    # Chronological segments used to measure importance agreement across
    # different portions of history ("stability across validation windows").
    consistency_windows: int = 2

    # Holdout-horizon fractions compared to measure "stability across
    # forecasting horizons" — a short holdout and the full configured
    # holdout, scored the same way as the window-consistency check but
    # varying prediction length rather than training data.
    horizon_fractions: tuple[float, ...] = (0.5, 1.0)

    # A feature's importance-weighted correlation with the target must clear
    # this magnitude to count as "business plausible".
    min_correlation_for_plausibility: float = 0.05

    # Observations required in a chronological segment before it is used in
    # a consistency comparison.
    min_observations_per_window: int = 8

    # A feature must hold at least this share of total importance to be
    # reported as the "dominant feature" for a given computation.
    dominant_feature_share_threshold: float = 0.3

    # An importance share moving by more than this between the point
    # estimate and any bootstrap/window/horizon vector is flagged as a
    # "significant" change in `StabilityMetrics.significant_changes`.
    significant_change_threshold: float = 0.25

    # Per-row local SHAP attributions are capped at this many rows (most
    # recent first) so a large curated dataset cannot bloat the stored
    # explainability report.
    max_local_importance_rows: int = 20

    # Sub-weights combining stability, plausibility and window-consistency
    # into one composite explainability score — the value Ranking consumes.
    stability_weight: float = 0.4
    plausibility_weight: float = 0.3
    window_consistency_weight: float = 0.3

    # Score given to a model with no permutable/explainable feature at all
    # (a purely univariate statistical fit, e.g. ARIMA). Neutral rather than
    # penalizing a capability the model was never given data to exercise.
    univariate_fallback_score: float = 1.0

    # Return the standard explainability configuration
    @classmethod
    def default(cls) -> "ExplainabilityConfig":
        return cls()

    # Build from a flat mapping, ignoring unknown keys
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExplainabilityConfig":
        if not payload:
            return cls()
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in payload.items() if key in allowed}
        if "horizon_fractions" in values:
            values["horizon_fractions"] = tuple(values["horizon_fractions"])
        return cls(**values)
