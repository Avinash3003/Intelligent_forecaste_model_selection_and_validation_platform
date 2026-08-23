"""TFT is selectable in the picker but never actually executed.

A product decision, not a bug: TFT needs torch and pytorch-forecasting
(~900 MB) which the Serverless environment does not install, so it cannot
run there — but the picker still offers it. The selection is dropped when
the execution request is built, so no runner or engine code has to know.

These tests pin both halves, because either one alone is wrong: if the strip
disappears the run fails on a missing dependency, and if the availability
block comes back the picker greys it out again.
"""

from __future__ import annotations

from app.config.model_availability import (
    CANDIDATE_MODEL_IDS,
    SILENTLY_SKIPPED_MODELS,
    filter_available,
    is_model_available,
    strip_silently_skipped,
    unavailable_models,
)


# --- offered: the picker must show TFT as selectable -------------------


def test_tft_is_offered_in_every_mode():
    for mode in ("local", "databricks", None, "  DataBricks "):
        assert unavailable_models(mode) == {}
        assert is_model_available("tft", mode) is True


def test_tft_survives_the_picker_filter():
    models = ["arima", "tft", "lightgbm"]
    assert filter_available(models, "databricks") == models


def test_tft_is_still_a_candidate_id():
    assert "tft" in CANDIDATE_MODEL_IDS


# --- not executed: it is dropped before the run --------------------------


def test_tft_is_stripped_from_a_selection():
    assert strip_silently_skipped(["prophet", "tft", "arima"]) == ["prophet", "arima"]


def test_stripping_preserves_order_of_the_rest():
    assert strip_silently_skipped(["xgboost", "tft", "arima", "prophet"]) == [
        "xgboost",
        "arima",
        "prophet",
    ]


def test_case_and_whitespace_do_not_smuggle_tft_through():
    assert strip_silently_skipped(["  TFT  ", "arima"]) == ["arima"]


def test_no_explicit_selection_stays_none():
    """None means 'use the engine's defaults'. Turning it into [] would
    train nothing at all."""
    assert strip_silently_skipped(None) is None


def test_a_tft_only_selection_collapses_to_no_explicit_selection():
    """The caller does `strip(...) or None`, so selecting only TFT falls back
    to the engine's defaults rather than submitting an empty model list."""
    assert (strip_silently_skipped(["tft"]) or None) is None


def test_a_selection_without_tft_is_untouched():
    models = ["prophet", "arima", "lightgbm", "xgboost"]
    assert strip_silently_skipped(models) == models


def test_the_skip_list_is_exactly_tft():
    """A second entry here would silently disable another model — that must
    be a deliberate edit, never a drift."""
    assert SILENTLY_SKIPPED_MODELS == frozenset({"tft"})
