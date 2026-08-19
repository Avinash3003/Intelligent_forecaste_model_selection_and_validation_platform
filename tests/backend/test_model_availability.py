"""A model is only offered when the execution mode can actually run it.

The Serverless job's environment spec deliberately omits torch and
pytorch-forecasting, so TFT is Unavailable there. Before this, the picker
offered it anyway and the estimator charged its per-fit weight for work
that never happened.
"""

from app.config.model_availability import (
    CANDIDATE_MODEL_IDS,
    filter_available,
    is_model_available,
    unavailable_models,
)


def test_serverless_cannot_run_tft():
    blocked = unavailable_models("databricks")
    assert "tft" in blocked
    assert "torch" in blocked["tft"]


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
    assert "tft" in unavailable_models("  DataBricks ")


def test_none_defaults_to_local():
    assert unavailable_models(None) == {}


def test_filter_available_preserves_order():
    models = ["arima", "tft", "lightgbm"]
    assert filter_available(models, "databricks") == ["arima", "lightgbm"]
    assert filter_available(models, "local") == models


def test_is_model_available_reads_the_same_table():
    assert is_model_available("tft", "local") is True
    assert is_model_available("tft", "databricks") is False
    assert is_model_available("arima", "databricks") is True


def test_candidate_ids_exclude_the_fallback():
    """seasonal_naive is the fallback, never a candidate to compete."""
    assert "seasonal_naive" not in CANDIDATE_MODEL_IDS
    assert set(CANDIDATE_MODEL_IDS) == {"prophet", "arima", "lightgbm", "xgboost", "tft"}
