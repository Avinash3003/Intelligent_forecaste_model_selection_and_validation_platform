"""reference_window.py — the shared window formula forward validation and
drift both consume.

window = min(history_length, max(2*seasonal_period, 3*horizon, min_periods))
"""

from __future__ import annotations

import numpy as np
import pytest

from forecast_engine.s06_evaluation.reference_window import (
    DEFAULT_MIN_REFERENCE_PERIODS,
    reference_window_length,
    recent_reference_slice,
    seasonal_period_for_frequency,
)


# ---------------------------------------------------------------------
# Frequency -> seasonal period
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "frequency,expected_period",
    [("Daily", 7), ("Weekly", 52), ("Monthly", 12), ("Quarterly", 4), ("Yearly", 1)],
)
def test_seasonal_period_matches_frequency(frequency, expected_period):
    assert seasonal_period_for_frequency(frequency) == expected_period


def test_unknown_or_missing_frequency_falls_back_to_the_platform_default():
    assert seasonal_period_for_frequency(None) == 12
    assert seasonal_period_for_frequency("Irregular") == 12


# ---------------------------------------------------------------------
# Window length — the three structural terms
# ---------------------------------------------------------------------


def test_monthly_series_uses_two_years_not_twelve_points():
    # 2 seasonal cycles = 2*12 = 24, vs 3*horizon = 3*12 = 36 -> the horizon
    # term wins for a standard 12-month-ahead forecast.
    window = reference_window_length(history_length=500, frequency="Monthly", forecast_horizon=12)
    assert window == 36


def test_daily_series_does_not_use_the_same_raw_count_as_monthly():
    # Daily: 2*7=14 vs 3*30=90 -> 90. A daily series forecasting 30 days
    # ahead needs a much larger raw observation count than a monthly one
    # forecasting 12 months ahead, even though both are "reasonable" windows.
    daily_window = reference_window_length(history_length=2000, frequency="Daily", forecast_horizon=30)
    monthly_window = reference_window_length(history_length=2000, frequency="Monthly", forecast_horizon=12)
    assert daily_window == 90
    assert monthly_window == 36
    assert daily_window != monthly_window


def test_weekly_series_uses_its_own_seasonal_cycle():
    # 2*52=104 vs 3*8=24 -> the seasonal-cycle term wins for a short
    # 8-week horizon.
    window = reference_window_length(history_length=1000, frequency="Weekly", forecast_horizon=8)
    assert window == 104


def test_window_never_exceeds_available_history():
    window = reference_window_length(history_length=20, frequency="Monthly", forecast_horizon=12)
    assert window == 20  # would otherwise compute to 36, but only 20 exist


def test_short_history_degrades_to_using_everything_available():
    """Below the structural floor, the window equals total history -
    identical to full-history behaviour, not a special-cased branch."""
    window = reference_window_length(history_length=13, frequency="Monthly", forecast_horizon=12)
    assert window == 13


def test_very_long_history_is_still_capped_by_the_structural_floor():
    # A 40-year monthly series (480 points) must not pull in the whole
    # thing - only the structurally-justified window.
    window = reference_window_length(history_length=480, frequency="Monthly", forecast_horizon=12)
    assert window == 36
    assert window < 480


def test_min_periods_floor_is_respected_even_for_a_tiny_horizon():
    # 2*4=8, 3*1=3, but min_periods=12 must still win.
    window = reference_window_length(history_length=100, frequency="Quarterly", forecast_horizon=1)
    assert window == DEFAULT_MIN_REFERENCE_PERIODS


def test_window_is_never_negative():
    assert reference_window_length(history_length=0, frequency="Monthly", forecast_horizon=12) == 0
    assert reference_window_length(history_length=5, frequency="Monthly", forecast_horizon=-3) >= 0


# ---------------------------------------------------------------------
# Slicing — does not invent data, preserves order
# ---------------------------------------------------------------------


def test_slice_returns_the_most_recent_values_in_original_order():
    history = np.arange(100, dtype=float)  # 0..99, chronological
    sliced = recent_reference_slice(history, "Monthly", forecast_horizon=12)
    assert len(sliced) == 36
    assert list(sliced) == list(range(64, 100))  # last 36, in order


def test_slice_never_invents_values_beyond_what_exists():
    history = np.arange(10, dtype=float)
    sliced = recent_reference_slice(history, "Monthly", forecast_horizon=12)
    assert len(sliced) == 10
    assert list(sliced) == list(range(10))
