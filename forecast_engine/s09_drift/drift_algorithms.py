"""The drift statistics themselves.

Each compares a reference distribution (history) against a current one (the
forward forecast) and returns one non-negative number — larger means more
drift. Registered in DRIFT_ALGORITHMS so nothing needs an if/elif chain to
dispatch, and adding a fifth is one registry entry.
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

    # `edges` covers only the reference's own [min, max]: np.histogram
    # silently drops any `current` value outside that range rather than
    # counting it, which the Laplace smoothing below then reads as
    # "uniform, no signal" — understating PSI for the most extreme
    # forecasts, the one case this statistic most needs to catch.
    #
    # Fixed here with a single open bin on each side (edges[0] -> -inf,
    # edges[-1] -> +inf) rather than graduated tail bins that grow with
    # how far out `current` reaches. A graduated approach was tried and
    # rejected: Laplace smoothing's denominator is `count + total_bins`,
    # so adding more bins to cover a MORE extreme value shrinks that
    # value's own smoothed proportion — the exact same raw concentration
    # (e.g. all 12 forecast points in one bin) reads as a WEAKER signal
    # the more bins were needed to reach it, understating the most
    # extreme forecasts even more than the original bug did. A single
    # fixed open bin per side never grows, so it carries no such penalty:
    # it does not further rank "72" against "300" (both simply "outside
    # the reference's observed range", which is the most a bin-membership
    # statistic can honestly claim about either), but it never
    # UNDERSTATES an extreme value relative to an ordinary in-range one,
    # which is the actual bug being fixed. Finer magnitude discrimination
    # for continuous, well-sampled data is what KS/Wasserstein are for —
    # algorithm_selector.py already routes to them instead of PSI in
    # exactly that situation; PSI is only selected here for the small or
    # low-cardinality histories where that finer ranking is not
    # statistically meaningful in the first place.
    #
    # Reference-side binning is unaffected either way: nothing in
    # `reference` was ever beyond its own min/max to begin with.
    bin_edges = edges.copy()
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

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
