"""backend/app/services/reference_window.py - mirrors
forecast_engine/s06_evaluation/reference_window.py's formula exactly, for
confidence's forecast-stability input (result_service.py)."""

from __future__ import annotations

from app.services.reference_window import reference_window_length, recent_reference_slice


def test_monthly_matches_the_engines_formula():
    # 2*12=24 vs 3*12=36 -> 36, identical to the engine-side test.
    assert reference_window_length(500, "Monthly", 12) == 36


def test_daily_differs_from_monthly_for_the_same_history_length():
    daily = reference_window_length(2000, "Daily", 30)
    monthly = reference_window_length(2000, "Monthly", 12)
    assert daily == 90
    assert monthly == 36
    assert daily != monthly


def test_never_exceeds_available_history():
    assert reference_window_length(20, "Monthly", 12) == 20


def test_short_history_degrades_to_everything_available():
    assert reference_window_length(13, "Monthly", 12) == 13


def test_unknown_frequency_falls_back_to_the_default_period():
    assert reference_window_length(500, None, 12) == 36
    assert reference_window_length(500, "Irregular", 12) == 36


def test_slice_preserves_order_and_never_invents_values():
    values = [float(i) for i in range(100)]
    sliced = recent_reference_slice(values, "Monthly", 12)
    assert len(sliced) == 36
    assert sliced == values[-36:]

    short_values = [1.0, 2.0, 3.0]
    assert recent_reference_slice(short_values, "Monthly", 12) == short_values
