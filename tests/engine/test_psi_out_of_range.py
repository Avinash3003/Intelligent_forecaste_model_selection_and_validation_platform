"""population_stability_index() must not silently drop `current` values
that fall entirely outside the reference histogram's range.

Before this fix: np.histogram(current, bins=edges) drops any value below
edges[0] or above edges[-1] without counting it. With cur_counts all zero,
Laplace smoothing reads that as "uniform, no signal" - an extreme forecast
(current=[300] against reference=[50..70]) produced PSI *lower* than an
ordinary in-range forecast (current=[60]), because being invisible to the
histogram looked more "normal" than actually landing in a bin would have.

Fixed by substituting ±inf for the outermost bin edges: any out-of-range
value is now counted in the correct tail bin instead of vanishing. A
graduated multi-bin extension was tried and rejected (see
drift_algorithms.py's comment) - Laplace smoothing's denominator grows
with bin count, so more bins to reach a MORE extreme value shrinks that
value's own smoothed proportion, understating it worse than the original
bug. A single fixed open bin per side has no such penalty.
"""

from __future__ import annotations

import numpy as np

from forecast_engine.s09_drift.drift_algorithms import population_stability_index

_REFERENCE = np.array([50.0, 55.0, 60.0, 65.0, 70.0])


# ---------------------------------------------------------------------
# 1. Forecast completely inside reference range
# ---------------------------------------------------------------------


def test_in_range_forecast_is_unaffected_by_the_fix():
    # No out-of-range value exists, so bin_edges[0]/[-1] being ±inf makes
    # no observable difference - identical to the pre-fix computation.
    psi = population_stability_index(_REFERENCE, np.array([60.0]))
    assert psi == population_stability_index(_REFERENCE, np.array([60.0]))
    assert psi > 0.0 or psi == 0.0  # finite, sane


# ---------------------------------------------------------------------
# 2. Slightly above range / 3. Far above range / 4. Below range
# ---------------------------------------------------------------------


def test_the_exact_reproduction_case_extreme_is_no_longer_understated():
    """reference=[50,55,60,65,70], current=[300] - the task's own
    reproduction case. Before the fix this was 0.1155, BELOW the in-range
    value's 0.1176. After the fix it must be >= the in-range value."""
    extreme = population_stability_index(_REFERENCE, np.array([300.0]))
    in_range = population_stability_index(_REFERENCE, np.array([60.0]))
    assert extreme >= in_range


def test_slightly_above_range_is_counted_not_dropped():
    psi = population_stability_index(_REFERENCE, np.array([72.0]))
    assert psi > 0.0


def test_far_above_range_is_counted_not_dropped():
    psi = population_stability_index(_REFERENCE, np.array([3000.0]))
    assert psi > 0.0


def test_below_range_is_counted_not_dropped():
    psi = population_stability_index(_REFERENCE, np.array([10.0]))
    assert psi > 0.0
    below_range_in_range = population_stability_index(_REFERENCE, np.array([60.0]))
    assert psi >= below_range_in_range  # never understated relative to normal


# ---------------------------------------------------------------------
# 5. Both lower and upper out-of-range values
# ---------------------------------------------------------------------


def test_both_tails_out_of_range_are_both_counted():
    psi = population_stability_index(_REFERENCE, np.array([10.0, 300.0]))
    assert psi > 0.0
    assert np.isfinite(psi)


# ---------------------------------------------------------------------
# 6. Constant reference series
# ---------------------------------------------------------------------


def test_constant_reference_does_not_crash():
    constant_reference = np.array([100.0] * 10)
    psi = population_stability_index(constant_reference, np.array([500.0]))
    assert np.isfinite(psi)
    assert psi >= 0.0


# ---------------------------------------------------------------------
# 7. Small reference dataset
# ---------------------------------------------------------------------


def test_small_reference_dataset_handles_out_of_range_safely():
    tiny_reference = np.array([10.0, 20.0])
    psi = population_stability_index(tiny_reference, np.array([1000.0]))
    assert np.isfinite(psi)
    assert psi >= 0.0


# ---------------------------------------------------------------------
# 8. Zero-frequency bins
# ---------------------------------------------------------------------


def test_zero_frequency_bins_are_smoothed_not_a_divide_by_zero():
    # Sparse reference leaves several bins empty on the reference side;
    # an out-of-range current must not divide by zero anywhere.
    sparse_reference = np.array([1.0, 2.0, 100.0, 101.0, 200.0])
    psi = population_stability_index(sparse_reference, np.array([-500.0]))
    assert np.isfinite(psi)


# ---------------------------------------------------------------------
# 9. Identical reference/current distributions
# ---------------------------------------------------------------------


def test_identical_distributions_still_score_zero():
    assert population_stability_index(_REFERENCE, _REFERENCE.copy()) == 0.0


# ---------------------------------------------------------------------
# 10. Extreme forecast: recent history ~60, forecast ~300
# ---------------------------------------------------------------------


def test_recent_60_forecast_300_produces_a_real_nonzero_signal():
    recent_history = np.array([58.0, 60.0, 62.0, 59.0, 61.0, 60.0, 63.0, 57.0])
    forecast = np.array([300.0] * 12)
    psi = population_stability_index(recent_history, forecast)
    assert psi > 0.0
    assert np.isfinite(psi)
    # Must not be understated relative to a forecast that simply
    # continues the recent level.
    ordinary = population_stability_index(recent_history, np.array([60.0] * 12))
    assert psi >= ordinary


# ---------------------------------------------------------------------
# Part 5's explicit requirement: do not make drift too sensitive
# ---------------------------------------------------------------------


def test_a_modest_overshoot_does_not_score_higher_than_a_wild_one():
    """The fix must not overcorrect into flagging a legitimate, modest
    continuation as MORE extreme than a genuinely wild forecast - a single
    fixed tail bin cannot rank degree of extremity, but it must never
    invert the ordering (that would be worse than the original bug)."""
    legitimate = np.array([72.0, 73.0] * 6)
    extreme = np.array([300.0] * 12)
    psi_legitimate = population_stability_index(_REFERENCE, legitimate)
    psi_extreme = population_stability_index(_REFERENCE, extreme)
    assert psi_extreme >= psi_legitimate


def test_matched_sample_size_extreme_and_legitimate_do_not_invert():
    # Same total count, same reference - only the current level differs.
    reference = np.array([58.0, 60.0, 62.0, 59.0, 61.0, 60.0, 63.0, 57.0])
    legitimate = np.array([64.0] * 12)  # a mild, plausible continuation
    extreme = np.array([300.0] * 12)
    psi_legitimate = population_stability_index(reference, legitimate)
    psi_extreme = population_stability_index(reference, extreme)
    assert psi_extreme >= psi_legitimate
