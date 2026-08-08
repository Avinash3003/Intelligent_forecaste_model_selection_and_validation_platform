"""Drift Detection, Threshold Estimation and Drift Validation configuration
(Sections 6.7, 6.8, 6.9).

Nothing here is a global constant applied to every forecasting group. The
algorithm a group is judged with, and the threshold its drift statistic is
compared against, are both *derived at runtime* from that group's own
history — this file only declares the decision rules and method parameters
that derivation uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DriftAlgorithm(str, Enum):
    """Statistical tests the platform can select between (Section 6.7).

    No single algorithm suits every distribution, which is exactly why
    selection is dynamic:

      * POPULATION_STABILITY_INDEX — bin-based, robust for small samples and
        discrete/low-cardinality distributions.
      * KOLMOGOROV_SMIRNOV — nonparametric CDF comparison; the standard
        choice for a continuous, non-normal distribution with enough data.
      * WASSERSTEIN_DISTANCE — an earth-mover distance that behaves well on
        roughly unimodal/near-normal continuous distributions.
      * JENSEN_SHANNON_DIVERGENCE — bounded and stable even on small,
        non-normal samples where a CDF test is unreliable.
    """

    POPULATION_STABILITY_INDEX = "population_stability_index"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    WASSERSTEIN_DISTANCE = "wasserstein_distance"
    JENSEN_SHANNON_DIVERGENCE = "jensen_shannon_divergence"


class ThresholdMethod(str, Enum):
    """Dynamic threshold estimation methods (Section 6.8).

    AUTO defers to the lightweight, sample-size-driven decision logic in
    `ThresholdEstimationConfig`; the others force a specific method.
    """

    AUTO = "auto"
    BOOTSTRAP = "bootstrap"
    PERCENTILE = "percentile"
    KERNEL_DENSITY = "kernel_density"


class NormalityTest(str, Enum):
    """Normality tests the algorithm selector can use."""

    SHAPIRO_WILK = "shapiro_wilk"
    DAGOSTINO_PEARSON = "dagostino_pearson"


@dataclass(frozen=True)
class AlgorithmSelectionConfig:
    """Drives Dynamic Drift Algorithm Selection (Section 6.7).

    The decision considers, in order: whether there is enough history to
    trust any distributional test, whether the distribution is effectively
    discrete/low-cardinality, and — only once both of those are settled —
    whether the history is approximately normal.
    """

    # Below this many observations, no distributional test is trusted; PSI's
    # binning degrades the most gracefully with limited data.
    min_sample_size: int = 20

    # A history with fewer distinct values than this, or whose unique/total
    # ratio falls below the ratio threshold, is treated as discrete-like.
    low_cardinality_unique_count: int = 8
    low_cardinality_unique_ratio: float = 0.15

    # Shapiro-Wilk is the more powerful test for small-to-moderate samples,
    # but scipy's implementation is only recommended up to a few thousand
    # observations; D'Agostino-Pearson is used above that size.
    shapiro_max_sample_size: int = 2000
    normality_alpha: float = 0.05

    # A non-normal distribution needs at least this many observations before
    # the Kolmogorov-Smirnov test's asymptotic behaviour is trustworthy;
    # below it, Jensen-Shannon divergence is used instead.
    min_sample_size_for_ks: int = 30


@dataclass(frozen=True)
class ThresholdEstimationConfig:
    """Drives Dynamic Threshold Estimation (Section 6.8).

    Every method answers the same question — "how large a drift statistic
    is normal even when nothing has actually drifted?" — by building a null
    distribution of the statistic from the group's own history and reading
    off a configured percentile of it.
    """

    method: ThresholdMethod = ThresholdMethod.AUTO

    # Percentile of the null distribution used as the threshold: a drift
    # statistic above it is judged too large to be historical noise.
    target_percentile: float = 95.0

    # AUTO method selection, by history length. Percentile estimation is the
    # cheapest and most robust with very little data; bootstrap needs enough
    # observations to split meaningfully; kernel density estimation is only
    # used once there is enough data to support a smooth density fit.
    min_size_for_bootstrap: int = 24
    min_size_for_kde: int = 60

    # Split-half resamples used by the bootstrap and kernel-density methods
    # to build the null distribution of the drift statistic.
    bootstrap_iterations: int = 200

    # Bins used by the histogram-based algorithms (PSI, Jensen-Shannon).
    histogram_bins: int = 10

    # Synthetic samples drawn from the fitted density for the kernel-density
    # method's percentile read-off.
    kde_sample_size: int = 2000

    random_state: int = 42


@dataclass(frozen=True)
class DriftValidationConfig:
    """Root configuration for drift detection and validation (Sections
    6.7–6.9).
    """

    algorithm_selection: AlgorithmSelectionConfig = field(default_factory=AlgorithmSelectionConfig)
    threshold_estimation: ThresholdEstimationConfig = field(default_factory=ThresholdEstimationConfig)

    # Return the standard drift validation configuration
    @classmethod
    def default(cls) -> "DriftValidationConfig":
        return cls()

    # Build from a nested mapping, ignoring unknown keys
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DriftValidationConfig":
        return cls(
            algorithm_selection=_build_block(
                AlgorithmSelectionConfig, payload.get("algorithm_selection")
            ),
            threshold_estimation=_build_block(
                ThresholdEstimationConfig, payload.get("threshold_estimation")
            ),
        )


# Instantiate one config block, keeping only keys it declares
def _build_block(block_type: type, values: dict[str, Any] | None) -> Any:
    if not values:
        return block_type()

    allowed = {f.name for f in block_type.__dataclass_fields__.values()}
    return block_type(**{key: value for key, value in values.items() if key in allowed})
