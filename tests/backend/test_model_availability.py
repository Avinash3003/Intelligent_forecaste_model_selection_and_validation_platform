"""What the picker offers, per execution mode.

Nothing is currently blocked: TFT is deliberately still offered even where
its dependencies are absent, and is dropped at submission instead — see
tests/backend/test_tft_offered_but_not_executed.py for that half. This table
remains the mechanism for genuinely hiding a model, so these tests keep its
behaviour pinned for the next thing that needs it.
"""

from app.config.model_availability import (
    CANDIDATE_MODEL_IDS,
    filter_available,
    is_model_available,
    unavailable_models,
)


def test_nothing_is_currently_blocked_on_serverless():
    assert unavailable_models("databricks") == {}


def test_serverless_can_run_every_other_candidate():
    blocked = unavailable_models("databricks")
    for model in ("arima", "prophet", "xgboost", "lightgbm"):
        assert model not in blocked


def test_local_can_run_everything():
    assert unavailable_models("local") == {}


def test_an_unknown_mode_is_assumed_capable():
    """A new execution mode is capable until something is known missing."""
    assert unavailable_models("some_future_mode") == {}


def test_mode_matching_is_case_and_whitespace_insensitive():
    """No entries today, but the lookup must still normalise the mode — a
    future block would otherwise miss "  DataBricks "."""
    assert unavailable_models("  DataBricks ") == unavailable_models("databricks")


def test_none_defaults_to_local():
    assert unavailable_models(None) == {}


def test_filter_available_preserves_order():
    models = ["arima", "tft", "lightgbm"]
    # Nothing is blocked, so every mode keeps the list intact and in order.
    assert filter_available(models, "databricks") == models
    assert filter_available(models, "local") == models


def test_is_model_available_reads_the_same_table():
    assert is_model_available("tft", "local") is True
    assert is_model_available("tft", "databricks") is True
    assert is_model_available("arima", "databricks") is True


def test_candidate_ids_exclude_the_fallback():
    """seasonal_naive is the fallback, never a candidate to compete."""
    assert "seasonal_naive" not in CANDIDATE_MODEL_IDS
    assert set(CANDIDATE_MODEL_IDS) == {"prophet", "arima", "lightgbm", "xgboost", "tft"}
