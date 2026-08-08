"""Dynamic Threshold Estimation (Section 6.8).

Answers one question for every forecasting group, independently: "how large
a drift statistic is normal even when nothing has actually drifted?" A null
distribution of the statistic is built from the group's own history — by
comparing historical sub-segments against each other, never against another
group's data — and the configured percentile of that null distribution
becomes the threshold.

Two registries drive the three supported methods:

  * `_NULL_BUILDERS` — how the null distribution of statistic values is
    generated (a handful of deterministic historical splits for PERCENTILE;
    many bootstrap resamples for BOOTSTRAP and KERNEL_DENSITY).
  * `_READERS` — how a threshold value is read off that null distribution
    (a raw percentile, or a percentile of a Gaussian-KDE-smoothed version
    of it).

Adding a later method (Beta distribution fitting, Bayesian updating) is one
new enum value plus one entry in each registry — `estimate()` itself never
changes, which is the "without changing business logic" requirement.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import gaussian_kde

from forecast_engine.config.drift_config import DriftAlgorithm, ThresholdEstimationConfig, ThresholdMethod
from forecast_engine.s09_drift.drift_algorithms import DRIFT_ALGORITHMS, DriftStatistic
from forecast_engine.s09_drift.drift_report import ThresholdEstimate

NullBuilder = Callable[[np.ndarray, DriftStatistic, ThresholdEstimationConfig, np.random.Generator], list[float]]
ThresholdReader = Callable[[list[float], ThresholdEstimationConfig, np.random.Generator], float]


# A handful of deterministic, non-overlapping historical splits
def _percentile_null(
    history: np.ndarray, statistic_fn: DriftStatistic, config: ThresholdEstimationConfig, rng: np.random.Generator
) -> list[float]:
    # Cheapest of the three: no resampling, so it stays usable on the
    # shortest histories the platform will still attempt to validate.
    chunk_count = max(2, min(4, history.size // 6))
    chunks = [chunk for chunk in np.array_split(history, chunk_count) if chunk.size > 0]

    samples = []
    for i in range(len(chunks) - 1):
        samples.append(statistic_fn(chunks[i], chunks[i + 1]))
    return samples


# Many resampled split-half comparisons of history against itself
def _bootstrap_null(
    history: np.ndarray, statistic_fn: DriftStatistic, config: ThresholdEstimationConfig, rng: np.random.Generator
) -> list[float]:
    # Standard bootstrap construction of a statistic's null distribution:
    # resample with replacement, split in half, score the two halves against
    # each other. Repeating this many times characterizes how large the
    # statistic gets from sampling noise alone.
    n = history.size
    half = max(1, n // 2)
    samples = []

    for _ in range(config.bootstrap_iterations):
        resampled = rng.choice(history, size=n, replace=True)
        a, b = resampled[:half], resampled[half:]
        if a.size and b.size:
            samples.append(statistic_fn(a, b))
    return samples


# Read a raw percentile off the null samples
def _percentile_reader(samples: list[float], config: ThresholdEstimationConfig, rng: np.random.Generator) -> float:
    return float(np.percentile(samples, config.target_percentile))


# Fit a Gaussian KDE and read the target percentile off a synthetic draw
def _kde_reader(samples: list[float], config: ThresholdEstimationConfig, rng: np.random.Generator) -> float:
    # The smoothing is the point: with enough null samples to support a
    # density fit, this interpolates between the discrete bootstrap values
    # rather than reading a raw (and therefore lumpy) empirical percentile.
    unique_values = np.unique(samples)
    if unique_values.size < 2:
        return float(unique_values[0]) if unique_values.size else 0.0

    density = gaussian_kde(samples)
    synthetic = density.resample(config.kde_sample_size, seed=rng)[0]
    return float(np.percentile(synthetic, config.target_percentile))


_NULL_BUILDERS: dict[ThresholdMethod, NullBuilder] = {
    ThresholdMethod.PERCENTILE: _percentile_null,
    ThresholdMethod.BOOTSTRAP: _bootstrap_null,
    ThresholdMethod.KERNEL_DENSITY: _bootstrap_null,
}

_READERS: dict[ThresholdMethod, ThresholdReader] = {
    ThresholdMethod.PERCENTILE: _percentile_reader,
    ThresholdMethod.BOOTSTRAP: _percentile_reader,
    ThresholdMethod.KERNEL_DENSITY: _kde_reader,
}


class ThresholdEstimator:
    """Estimates a drift threshold for one forecasting group's history."""

    # Store the threshold estimation config
    def __init__(self, config: ThresholdEstimationConfig | None = None) -> None:
        self._config = config or ThresholdEstimationConfig()

    # Estimate the drift threshold for history, judged with this algorithm
    def estimate(self, algorithm: DriftAlgorithm, history: np.ndarray) -> ThresholdEstimate:
        method = self._select_method(history.size)
        statistic_fn = DRIFT_ALGORITHMS[algorithm]
        rng = np.random.default_rng(self._config.random_state)

        null_samples = _NULL_BUILDERS[method](history, statistic_fn, self._config, rng)

        # A history too small to build any null comparison yields a
        # threshold of 0.0 rather than raising — Drift Validation then
        # treats any nonzero statistic as drift, the conservative behaviour
        # for evidence this thin.
        if not null_samples:
            return ThresholdEstimate(
                method=method,
                value=0.0,
                reason="History too short to build a null distribution of the drift statistic.",
                null_sample_size=0,
            )

        value = _READERS[method](null_samples, self._config, rng)

        return ThresholdEstimate(
            method=method,
            value=max(0.0, value),
            reason=(
                f"{method.value} threshold at the {self._config.target_percentile:.0f}th percentile of "
                f"{len(null_samples)} null comparison(s) drawn from this group's own history."
            ),
            null_sample_size=len(null_samples),
        )

    # Lightweight, sample-size-driven method selection
    def _select_method(self, sample_size: int) -> ThresholdMethod:
        # Percentile needs the least data and is used below the bootstrap
        # floor; bootstrap needs enough observations that resampled halves
        # are still meaningful; KDE is reserved for histories with enough
        # data to support a smooth fit.
        if self._config.method is not ThresholdMethod.AUTO:
            return self._config.method

        if sample_size < self._config.min_size_for_bootstrap:
            return ThresholdMethod.PERCENTILE
        if sample_size < self._config.min_size_for_kde:
            return ThresholdMethod.BOOTSTRAP
        return ThresholdMethod.KERNEL_DENSITY
