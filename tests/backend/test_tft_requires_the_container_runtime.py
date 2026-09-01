"""A model the user picked is either run or refused — never dropped.

TFT needs torch and pytorch-forecasting, which only the ForecastIQ
container image carries. It used to be stripped from `selected_models`
silently, so choosing it changed nothing about a run: it never appeared in
the results, the comparison table or MLflow, and nothing anywhere said why.
That is indistinguishable from a model that ran and lost.

Now the platform runs it on container compute and refuses it elsewhere,
naming the reason.
"""

from __future__ import annotations

import pytest

from app.config.model_availability import (
    CANDIDATE_MODEL_IDS,
    CONTAINER_ONLY_MODELS,
    unsupported_models,
)


# --- what the picker offers -------------------------------------------


def test_tft_is_still_offered_as_a_candidate():
    assert "tft" in CANDIDATE_MODEL_IDS


def test_tft_is_the_model_that_needs_the_container():
    assert CONTAINER_ONLY_MODELS == frozenset({"tft"})


# --- container compute runs it ----------------------------------------


def test_the_container_runtime_supports_every_candidate():
    """Nothing is unsupported on the image that installs the dependencies."""
    assert unsupported_models(list(CANDIDATE_MODEL_IDS), uses_container=True) == {}


def test_a_container_run_keeps_tft_in_the_selection():
    assert unsupported_models(["prophet", "tft"], uses_container=True) == {}


# --- other compute refuses it, with a reason --------------------------


def test_tft_on_non_container_compute_is_reported_unsupported():
    unsupported = unsupported_models(["prophet", "tft", "arima"], uses_container=False)

    assert set(unsupported) == {"tft"}
    assert "container runtime" in unsupported["tft"]
    assert "New Job Compute" in unsupported["tft"]


def test_the_other_models_are_never_reported_unsupported():
    assert unsupported_models(["prophet", "arima", "lightgbm", "xgboost"], uses_container=False) == {}


def test_the_check_is_case_and_whitespace_insensitive():
    assert set(unsupported_models(["  TFT  "], uses_container=False)) == {"  TFT  "}


def test_no_explicit_selection_is_left_alone():
    """None means "the engine's own defaults", which must stay meaningful."""
    assert unsupported_models(None, uses_container=False) == {}
    assert unsupported_models(None, uses_container=True) == {}


# --- the selection is never silently trimmed --------------------------


def test_nothing_in_the_module_strips_models_any_more():
    """The regression this file exists for: a silent trim looks exactly like
    a model that ran and was not selected."""
    import app.config.model_availability as availability

    assert not hasattr(availability, "strip_silently_skipped")
    assert not hasattr(availability, "SILENTLY_SKIPPED_MODELS")
