"""Shared reference-window selection for forward validation and drift
(Sections 6.5, 6.7-6.9).

Comparing a forecast against a series' ENTIRE history is only meaningful
when that history is a single, roughly consistent regime. A long series
that has genuinely moved to a new level (temperature over decades, gold or
crypto prices, even ordinary demand after a structural shift) makes "the
whole history" a misleading yardstick: a forecast that correctly continues
today's regime can look like an outlier purely because most of the
historical mass sits at an older, since-departed level.

This module answers one narrow question — "how many of the most recent
observations should be treated as the reference population?" — from
structural facts about the series (frequency, seasonal cycle length,
forecast horizon, how much history exists), never from the forecast being
judged or from how well any candidate happens to score. The same series
gets the same window regardless of which model produced the forecast.
"""

from __future__ import annotations

import numpy as np

# Periods per seasonal cycle, by detected frequency (frequency_detector.py's
# own vocabulary — Daily/Weekly/Monthly/Quarterly/Yearly). Daily assumes a
# weekly cycle (7), the most common sub-monthly seasonal pattern (retail,
# traffic, sensor data); Weekly a yearly cycle (52); Monthly/Quarterly their
# calendar cycle. Yearly and an undetected/Irregular frequency fall back to
# the platform's existing default (12): with no sub-annual cycle to size a
# window around, the horizon- and floor-based terms below do the real work
# instead.
_SEASONAL_PERIOD_BY_FREQUENCY: dict[str, int] = {
    "Daily": 7,
    "Weekly": 52,
    "Monthly": 12,
    "Quarterly": 4,
    "Yearly": 1,
}
_DEFAULT_SEASONAL_PERIOD = 12

# Minimum reference window regardless of frequency/horizon. Matches
# ForwardValidationConfig.min_history_for_validation — the floor below
# which the platform already treats a series as too short to validate.
DEFAULT_MIN_REFERENCE_PERIODS = 12

# Full seasonal cycles the window should cover when history allows it. Two
# is not a new number: MissingSeasonalityRule already requires two full
# cycles before it will judge seasonality at all (validation_rules.py) —
# reused here rather than invented.
SEASONAL_CYCLES = 2

# Multiple of the forecast horizon the reference population should be, at
# minimum, so a distributional comparison against it has reasonable power.
HORIZON_MULTIPLE = 3


def seasonal_period_for_frequency(frequency: str | None) -> int:
    """Periods per seasonal cycle for a detected frequency string.

    Unknown/Irregular/None frequencies fall back to the platform's existing
    default (12) rather than raising — a caller must never fail validation
    just because frequency detection was inconclusive.
    """
    if frequency is None:
        return _DEFAULT_SEASONAL_PERIOD
    return _SEASONAL_PERIOD_BY_FREQUENCY.get(frequency, _DEFAULT_SEASONAL_PERIOD)


def reference_window_length(
    history_length: int,
    frequency: str | None,
    forecast_horizon: int,
    min_periods: int = DEFAULT_MIN_REFERENCE_PERIODS,
) -> int:
    """How many of the most recent observations form the reference
    population for forward validation and drift.

        window = min(
            history_length,
            max(
                SEASONAL_CYCLES * seasonal_period,     # seasonal context
                HORIZON_MULTIPLE * forecast_horizon,    # a fair-power test
                min_periods,                            # existing floor
            ),
        )

    Deliberately not a function of measured trend/volatility: sizing the
    window from a property you would need the window to measure first is
    circular. Trend and regime are instead handled by WHAT gets compared
    within the window (its own recent level/spread), not by how large the
    window is.

    Never exceeds history_length, so a short series (already too little to
    usefully window) degrades to using everything available — identical to
    full-history behaviour, not a special case that needs its own branch.
    """
    seasonal_period = seasonal_period_for_frequency(frequency)
    floor = max(
        SEASONAL_CYCLES * seasonal_period,
        HORIZON_MULTIPLE * max(forecast_horizon, 0),
        min_periods,
    )
    return max(0, min(history_length, floor))


def recent_reference_slice(
    history: np.ndarray,
    frequency: str | None,
    forecast_horizon: int,
    min_periods: int = DEFAULT_MIN_REFERENCE_PERIODS,
) -> np.ndarray:
    """The most recent `reference_window_length(...)` observations of
    `history`, in original order — what a rule or drift check should treat
    as "what does this series look like right now," instead of the whole
    series.
    """
    window = reference_window_length(len(history), frequency, forecast_horizon, min_periods)
    if window <= 0:
        return history[:0]
    return history[-window:]
