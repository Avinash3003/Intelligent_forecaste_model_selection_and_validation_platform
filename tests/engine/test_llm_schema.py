"""Structured output validation (Section 13.1)."""

import json

import pytest

from forecast_engine.s11_llm.schema import (
    MAX_CAVEATS,
    MAX_REJECTION_REASONS,
    MAX_SUMMARY_WORDS,
    InsightPayload,
    SchemaValidationError,
    parse_and_validate,
)

VALID = {
    "selected_model": "xgboost",
    "rejection_reasons": ["prophet: missing seasonality"],
    "confidence": 74.1,
    "caveats": [],
    "concise_summary": "xgboost was selected, with a backtest WMAPE of 25.9%, meaning it was off by about 26% on average.",
}


def test_valid_payload_parses_cleanly():
    payload = parse_and_validate(json.dumps(VALID))
    assert payload.selected_model == "xgboost"
    assert payload.confidence == 74.1
    assert payload.rejection_reasons == ["prophet: missing seasonality"]


def test_markdown_fenced_json_is_unwrapped():
    fenced = f"```json\n{json.dumps(VALID)}\n```"
    payload = parse_and_validate(fenced)
    assert payload.selected_model == "xgboost"


def test_garbage_is_not_valid_json():
    with pytest.raises(SchemaValidationError, match="not valid JSON"):
        parse_and_validate("this is not json at all")


def test_json_array_instead_of_object_is_rejected():
    with pytest.raises(SchemaValidationError, match="must be an object"):
        parse_and_validate("[1, 2, 3]")


def test_missing_required_field_is_rejected():
    payload = dict(VALID)
    del payload["concise_summary"]
    with pytest.raises(SchemaValidationError, match="Missing required"):
        parse_and_validate(json.dumps(payload))


def test_empty_selected_model_is_rejected():
    payload = dict(VALID, selected_model="")
    with pytest.raises(SchemaValidationError, match="selected_model"):
        parse_and_validate(json.dumps(payload))


def test_wrong_selected_model_is_rejected_against_expected():
    with pytest.raises(SchemaValidationError, match="actual selected model"):
        parse_and_validate(json.dumps(VALID), expected_model="lightgbm")


def test_matching_expected_model_is_accepted_case_insensitively():
    payload = parse_and_validate(json.dumps(VALID), expected_model="XGBoost")
    assert payload.selected_model == "xgboost"


def test_confidence_out_of_range_is_rejected():
    payload = dict(VALID, confidence=150)
    with pytest.raises(SchemaValidationError, match="confidence"):
        parse_and_validate(json.dumps(payload))


def test_confidence_null_is_accepted():
    payload = dict(VALID, confidence=None)
    result = parse_and_validate(json.dumps(payload))
    assert result.confidence is None


def test_confidence_as_string_number_is_coerced():
    payload = dict(VALID, confidence="74.1")
    result = parse_and_validate(json.dumps(payload))
    assert result.confidence == 74.1


def test_too_many_rejection_reasons_is_rejected():
    payload = dict(VALID, rejection_reasons=[f"model{i}: reason" for i in range(MAX_REJECTION_REASONS + 1)])
    with pytest.raises(SchemaValidationError, match="rejection_reasons"):
        parse_and_validate(json.dumps(payload))


def test_too_many_caveats_is_rejected():
    payload = dict(VALID, caveats=["a"] * (MAX_CAVEATS + 1))
    with pytest.raises(SchemaValidationError, match="caveats"):
        parse_and_validate(json.dumps(payload))


def test_rejection_reasons_must_be_strings():
    payload = dict(VALID, rejection_reasons=[{"nested": "object"}])
    with pytest.raises(SchemaValidationError, match="rejection_reasons"):
        parse_and_validate(json.dumps(payload))


def test_oversized_summary_is_rejected():
    payload = dict(VALID, concise_summary=" ".join(["word"] * (MAX_SUMMARY_WORDS + 1)))
    with pytest.raises(SchemaValidationError, match="concise_summary"):
        parse_and_validate(json.dumps(payload))


def test_empty_summary_is_rejected():
    payload = dict(VALID, concise_summary="")
    with pytest.raises(SchemaValidationError, match="concise_summary"):
        parse_and_validate(json.dumps(payload))


def test_multiple_problems_are_all_reported_at_once():
    payload = {"selected_model": "", "rejection_reasons": "not a list", "confidence": 999, "caveats": [], "concise_summary": ""}
    with pytest.raises(SchemaValidationError) as excinfo:
        parse_and_validate(json.dumps(payload))
    # Every distinct problem surfaces, not just the first — a retry prompt
    # can fix all of them in one pass instead of discovering them one at a time.
    assert len(excinfo.value.problems) >= 3


def test_round_trip_to_dict_from_dict():
    payload = InsightPayload.from_dict(VALID)
    assert InsightPayload.from_dict(payload.to_dict()) == payload
