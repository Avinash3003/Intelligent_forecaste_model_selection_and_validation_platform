"""Drift statistic implementations (Section 6.7).

Each function compares a `reference` distribution (history) against a
`current` one (the forward forecast) and returns a single non-negative
statistic — larger means more drift. Registered in `DRIFT_ALGORITHMS` so the
selector and threshold estimator never need an if/elif chain to dispatch on
the chosen algorithm; adding a fifth algorithm later is one registry entry.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

from forecast_engine.config.drift_config import DriftAlgorithm

DriftStatistic = Callable[[np.ndarray, np.ndarray], float]


# PSI: distribution shift across bins from the reference's own quantiles
def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        # Too few distinct reference values for meaningful binning; fall
        # back to two bins split at the median.
        edges = np.unique([np.min(reference), np.median(reference), np.max(reference)])
    if edges.size < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Laplace smoothing avoids a divide-by-zero when a bin is empty on
    # either side — an unpopulated bin should count as "small", not crash.
    ref_prop = (ref_counts + 1) / (ref_counts.sum() + len(ref_counts))
    cur_prop = (cur_counts + 1) / (cur_counts.sum() + len(cur_counts))

    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


# KS statistic: maximum gap between the two empirical CDFs
def kolmogorov_smirnov(reference: np.ndarray, current: np.ndarray, **_: object) -> float:
    return float(stats.ks_2samp(reference, current).statistic)


# Earth-mover distance, normalized by the reference's own spread
def wasserstein_distance(reference: np.ndarray, current: np.ndarray, **_: object) -> float:
    raw = stats.wasserstein_distance(reference, current)
    scale = float(np.std(reference))
    # Normalizing makes the statistic a dimensionless ratio, comparable
    # across forecasting groups of different magnitude and units.
    if scale == 0:
        return float(raw)
    return float(raw / scale)


# Bounded [0, 1] divergence between histograms of the two samples
def jensen_shannon_divergence(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    combined_min = float(min(np.min(reference), np.min(current)))
    combined_max = float(max(np.max(reference), np.max(current)))
    if combined_min == combined_max:
        return 0.0

    edges = np.linspace(combined_min, combined_max, bins + 1)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_prop = (ref_counts + 1e-9) / (ref_counts.sum() + 1e-9 * bins)
    cur_prop = (cur_counts + 1e-9) / (cur_counts.sum() + 1e-9 * bins)

    divergence = jensenshannon(ref_prop, cur_prop, base=2)
    return float(divergence) if np.isfinite(divergence) else 0.0


DRIFT_ALGORITHMS: dict[DriftAlgorithm, DriftStatistic] = {
    DriftAlgorithm.POPULATION_STABILITY_INDEX: population_stability_index,
    DriftAlgorithm.KOLMOGOROV_SMIRNOV: kolmogorov_smirnov,
    DriftAlgorithm.WASSERSTEIN_DISTANCE: wasserstein_distance,
    DriftAlgorithm.JENSEN_SHANNON_DIVERGENCE: jensen_shannon_divergence,
}
