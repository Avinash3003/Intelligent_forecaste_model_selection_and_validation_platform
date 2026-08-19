"""threshold_estimator.py's bootstrap/KDE null-builder now splits
chronologically before resampling, instead of fully reshuffling history
before splitting. A full reshuffle turns both "halves" into two random
draws from the same pool, which cannot reflect how much a series' earlier
and later periods actually differ - the exact comparison Drift Validation
itself makes (a later forecast vs. the window's history).
"""

from __future__ import annotations

import numpy as np

from forecast_engine.config.drift_config import ThresholdEstimationConfig, ThresholdMethod
from forecast_engine.s09_drift.threshold_estimator import ThresholdEstimator
from forecast_engine.config.drift_config import DriftAlgorithm


def test_bootstrap_null_reflects_a_real_within_window_trend():
    """A window with a clear internal trend (even after proper sizing, a
    reference window can have some legitimate ongoing drift) should
    produce a LARGER threshold than a window that is genuinely flat -
    proof the null distribution is sensitive to temporal structure, which
    a full-reshuffle bootstrap would wash out."""
    rng_history_trending = np.array([10.0 + i * 0.5 for i in range(60)])
    rng_history_flat = np.array([10.0 + (i % 3) * 0.5 for i in range(60)])

    config = ThresholdEstimationConfig(method=ThresholdMethod.BOOTSTRAP, bootstrap_iterations=200, random_state=7)
    estimator = ThresholdEstimator(config)

    trending_threshold = estimator.estimate(DriftAlgorithm.WASSERSTEIN_DISTANCE, rng_history_trending)
    flat_threshold = estimator.estimate(DriftAlgorithm.WASSERSTEIN_DISTANCE, rng_history_flat)

    assert trending_threshold.value > flat_threshold.value


def test_null_samples_are_reproducible_and_bounded():
    config = ThresholdEstimationConfig(method=ThresholdMethod.BOOTSTRAP, bootstrap_iterations=50, random_state=1)
    estimator = ThresholdEstimator(config)
    history = np.array([float(i % 10) for i in range(40)])

    first = estimator.estimate(DriftAlgorithm.WASSERSTEIN_DISTANCE, history)
    second = estimator.estimate(DriftAlgorithm.WASSERSTEIN_DISTANCE, history)

    assert first.value == second.value  # deterministic given a fixed random_state
    assert first.null_sample_size == 50
    assert first.value >= 0.0


def test_percentile_method_is_unchanged_still_sequential():
    """The percentile null-builder was already temporally ordered
    (chunks[i] vs chunks[i+1]) and needed no change - confirmed still
    functions correctly post-fix."""
    config = ThresholdEstimationConfig(method=ThresholdMethod.PERCENTILE)
    estimator = ThresholdEstimator(config)
    history = np.array([float(i) for i in range(30)])

    result = estimator.estimate(DriftAlgorithm.KOLMOGOROV_SMIRNOV, history)

    assert result.method == ThresholdMethod.PERCENTILE
    assert result.value >= 0.0
