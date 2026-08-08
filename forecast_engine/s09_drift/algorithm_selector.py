"""Dynamic Drift Algorithm Selection (Section 6.7).

Chooses one of the four registered drift algorithms for a forecasting
group's history, based only on properties measurable from that history
itself: sample size, cardinality, and a normality test. No algorithm is
hardcoded per dataset or per model family — the same function looks at
every group's history exactly the same way and can land on a different
algorithm for each.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from forecast_engine.config.drift_config import AlgorithmSelectionConfig, DriftAlgorithm
from forecast_engine.s09_drift.drift_report import DriftAlgorithmSelection


class DriftAlgorithmSelector:
    """Selects the drift algorithm appropriate for one history sample."""

    # Store the selection thresholds config
    def __init__(self, config: AlgorithmSelectionConfig | None = None) -> None:
        self._config = config or AlgorithmSelectionConfig()

    def select(self, history: np.ndarray) -> DriftAlgorithmSelection:
        """Pick the drift algorithm this history is best suited to.

        The decision is a four-way tree, evaluated in this order — the first
        matching branch wins:

            too few samples ............................ PSI
            low cardinality (discrete-looking) ......... PSI
            normal distribution ........................ Wasserstein
            non-normal, enough samples for KS .......... Kolmogorov-Smirnov
            non-normal, too few for KS ................. Jensen-Shannon

        The order matters: sample size and cardinality are checked before the
        normality test because a normality test on a tiny or near-discrete
        sample is not meaningful in the first place.

        Every branch records its own `reason` in prose, which is what the
        dashboard and MLflow show — the selection is never reported as a bare
        algorithm name with no justification.
        """
        config = self._config
        n = int(history.size)
        unique_count = int(np.unique(history).size)
        unique_ratio = unique_count / n if n else 0.0

        # Shared across every branch below, so each branch only has to state
        # the algorithm and why it was chosen.
        def selection(algorithm: DriftAlgorithm, reason: str, **normality: object) -> DriftAlgorithmSelection:
            return DriftAlgorithmSelection(
                algorithm=algorithm,
                reason=reason,
                sample_size=n,
                unique_values=unique_count,
                **normality,
            )

        # 1. Too little data for any distributional test to mean much.
        if n < config.min_sample_size:
            return selection(
                DriftAlgorithm.POPULATION_STABILITY_INDEX,
                f"Sample size ({n}) is below the minimum ({config.min_sample_size}) for a "
                f"distributional test; PSI's bin-based approach remains stable with limited data.",
            )

        # 2. Values repeat enough that the series behaves like a discrete
        #    variable, where a continuous-distribution test does not apply.
        if unique_count < config.low_cardinality_unique_count or unique_ratio < config.low_cardinality_unique_ratio:
            return selection(
                DriftAlgorithm.POPULATION_STABILITY_INDEX,
                f"History has {unique_count} distinct value(s) ({unique_ratio:.2%} of observations), "
                f"a discrete/low-cardinality distribution better suited to bin-based PSI than a "
                f"continuous-distribution test.",
            )

        test_name, p_value = self._normality_test(history)
        normality = {"normality_test": test_name, "normality_p_value": p_value}

        # 3. Near-normal: Wasserstein's distance is expressed in the data's own
        #    units, which keeps the statistic comparable across groups.
        if p_value > config.normality_alpha:
            return selection(
                DriftAlgorithm.WASSERSTEIN_DISTANCE,
                f"History approximates a normal distribution ({test_name} p={p_value:.4f} > "
                f"{config.normality_alpha}); Wasserstein distance gives a scale-consistent drift "
                f"measure for near-normal data.",
                is_normal=True,
                **normality,
            )

        # 4. Non-normal with enough samples for KS's asymptotics to hold.
        if n >= config.min_sample_size_for_ks:
            return selection(
                DriftAlgorithm.KOLMOGOROV_SMIRNOV,
                f"History is non-normal ({test_name} p={p_value:.4f} <= {config.normality_alpha}) "
                f"with sufficient sample size ({n}) for a nonparametric CDF comparison.",
                is_normal=False,
                **normality,
            )

        # 5. Non-normal and small — JS stays bounded where KS would not be
        #    trustworthy.
        return selection(
            DriftAlgorithm.JENSEN_SHANNON_DIVERGENCE,
            f"History is non-normal ({test_name} p={p_value:.4f}) with fewer than "
            f"{config.min_sample_size_for_ks} observations; Jensen-Shannon divergence stays bounded "
            f"and stable for small, non-normal samples where the KS test's asymptotics are unreliable.",
            is_normal=False,
            **normality,
        )

    # Shapiro-Wilk below the sample-size cutoff, else D'Agostino-Pearson
    def _normality_test(self, history: np.ndarray) -> tuple[str, float]:
        if history.size <= self._config.shapiro_max_sample_size:
            _, p_value = stats.shapiro(history)
            return "Shapiro-Wilk", float(p_value)

        _, p_value = stats.normaltest(history)
        return "D'Agostino-Pearson", float(p_value)
