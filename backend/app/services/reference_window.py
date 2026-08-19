"""Backend mirror of forecast_engine/s06_evaluation/reference_window.py.

Duplicated deliberately, not imported: the backend and the engine are
separate deployable units (the engine runs as a subprocess locally, or as a
standalone Databricks job — see local_runner.py/databricks_runner.py), so
the backend never imports forecast_engine code directly. The formula this
mirrors is small (one comparison, three terms) and used in exactly one
place here (confidence's forecast-stability input in result_service.py),
so a second small, independently-testable copy of the same logic is a far
smaller footprint than adding a new field to the run summary schema just to
carry a single integer across the process boundary.

If this formula changes, update forecast_engine's copy to match.
"""

from __future__ import annotations

# Periods per seasonal cycle, by detected frequency (matches
# frequency_detector.py's vocabulary and forecast_engine's own copy of this
# table). Daily assumes a weekly cycle; Weekly a yearly cycle;
# Monthly/Quarterly their calendar cycle. Yearly/unknown fall back to the
# platform's existing default of 12.
_SEASONAL_PERIOD_BY_FREQUENCY: dict[str, int] = {
    "Daily": 7,
    "Weekly": 52,
    "Monthly": 12,
    "Quarterly": 4,
    "Yearly": 1,
}
_DEFAULT_SEASONAL_PERIOD = 12

# Matches forecast_engine's ForwardValidationConfig.min_history_for_validation.
DEFAULT_MIN_REFERENCE_PERIODS = 12
SEASONAL_CYCLES = 2
HORIZON_MULTIPLE = 3


def reference_window_length(
    history_length: int,
    frequency: str | None,
    forecast_horizon: int,
    min_periods: int = DEFAULT_MIN_REFERENCE_PERIODS,
) -> int:
    """How many of the most recent observations form the reference
    population — same formula as forecast_engine's copy:

        window = min(history_length, max(2*seasonal_period, 3*horizon, min_periods))

    Never exceeds history_length, so a short series degrades to using
    everything available rather than needing a special case.
    """
    seasonal_period = _SEASONAL_PERIOD_BY_FREQUENCY.get(frequency, _DEFAULT_SEASONAL_PERIOD)
    floor = max(
        SEASONAL_CYCLES * seasonal_period,
        HORIZON_MULTIPLE * max(forecast_horizon, 0),
        min_periods,
    )
    return max(0, min(history_length, floor))


def recent_reference_slice(
    values: list[float],
    frequency: str | None,
    forecast_horizon: int,
    min_periods: int = DEFAULT_MIN_REFERENCE_PERIODS,
) -> list[float]:
    """The most recent `reference_window_length(...)` values, in original
    order — what confidence's stability check should compare the forecast
    against, instead of the group's entire observed history."""
    window = reference_window_length(len(values), frequency, forecast_horizon, min_periods)
    if window <= 0:
        return []
    return values[-window:]
