"""The weights behind the composite ranking score.

Ranking combines three independent signals — normalized backtest metrics,
forecast stability, and SHAP/importance consistency — so it is never a rerun
of "lowest error wins". Every weight is a value here, not a literal in the
ranking code.

The SHAP signal is produced by the explainability stage before ranking runs;
ranking only owns how much it counts.

The three top-level weights are normalized at use time, so a deployment can
express relative emphasis without making them sum to 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricWeightsConfig:
    """Weight of each backtest metric inside the accuracy score.

    Each is min-max normalized across the group's candidates (lower error ->
    higher score) then combined. WMAPE weighs most, as the primary measure.
    """

    mape_weight: float = 0.15
    wmape_weight: float = 0.35
    rmse_weight: float = 0.2
    mae_weight: float = 0.15
    smape_weight: float = 0.15


@dataclass(frozen=True)
class StabilityWeightsConfig:
    """Weight of each stability signal.

    Stability judges the shape of the forward forecast, independent of
    backtest accuracy: a model can score well on history and still ship an
    erratic forecast.
    """

    variance_weight: float = 0.4
    smoothness_weight: float = 0.35
    interval_weight: float = 0.25

    # Score assigned to a model that produces no confidence intervals
    # (every tree model in this phase). Neutral rather than punitive: the
    # model is not unstable, it simply was not asked to quantify uncertainty.
    no_interval_score: float = 0.5


@dataclass(frozen=True)
class CompositeWeightsConfig:
    """Top-level weights combining the three ranking components.

    Normalized at use time, so these express *relative* emphasis rather
    than needing to sum to exactly 1.0.
    """

    backtest_weight: float = 0.5
    stability_weight: float = 0.25
    shap_weight: float = 0.25


@dataclass(frozen=True)
class RankingConfig:
    """Root configuration for Model Ranking (Section 6.6)."""

    metric_weights: MetricWeightsConfig = field(default_factory=MetricWeightsConfig)
    stability_weights: StabilityWeightsConfig = field(default_factory=StabilityWeightsConfig)
    composite_weights: CompositeWeightsConfig = field(default_factory=CompositeWeightsConfig)

    # Return the standard ranking configuration
    @classmethod
    def default(cls) -> "RankingConfig":
        return cls()

    # Build from a nested mapping, ignoring unknown keys
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RankingConfig":
        return cls(
            metric_weights=_build_block(MetricWeightsConfig, payload.get("metric_weights")),
            stability_weights=_build_block(StabilityWeightsConfig, payload.get("stability_weights")),
            composite_weights=_build_block(CompositeWeightsConfig, payload.get("composite_weights")),
        )


# Instantiate one config block, keeping only keys it declares
def _build_block(block_type: type, values: dict[str, Any] | None) -> Any:
    if not values:
        return block_type()

    allowed = {f.name for f in block_type.__dataclass_fields__.values()}
    return block_type(**{key: value for key, value in values.items() if key in allowed})
