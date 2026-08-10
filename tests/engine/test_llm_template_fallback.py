"""The non-LLM deterministic fallback (Section 11)."""

from forecast_engine.s11_llm.grounding import check_grounding
from forecast_engine.s11_llm.template_fallback import build_template_insight


def test_a_clean_win_produces_a_grounded_summary():
    payload = build_template_insight(
        selected_model="xgboost", wmape=25.86, is_fallback=False, fallback_trigger=None,
        rejected_candidates=[], confidence_pct=74.1, caveats=[],
    )
    assert payload.selected_model == "xgboost"
    assert "25.86" in payload.concise_summary
    result = check_grounding(payload.concise_summary, {"wmape": 25.86})
    assert result.grounded


def test_a_fallback_states_it_plainly_and_cites_the_trigger():
    payload = build_template_insight(
        selected_model="seasonal_naive", wmape=None, is_fallback=True,
        fallback_trigger="All 1 evaluated model(s) failed validation (xgboost).",
        rejected_candidates=[{"model_name": "xgboost", "reason": "drift statistic exceeded threshold"}],
        confidence_pct=94.0, caveats=["fallback model used"],
    )
    assert "fallback" in payload.concise_summary.lower()
    assert payload.rejection_reasons == ["xgboost: drift statistic exceeded threshold"]
    assert "fallback model used" in payload.caveats


def test_rejection_reasons_are_capped():
    rejected = [{"model_name": f"model{i}", "reason": "eliminated"} for i in range(10)]
    payload = build_template_insight(
        selected_model="xgboost", wmape=10.0, is_fallback=False, fallback_trigger=None,
        rejected_candidates=rejected, confidence_pct=90.0, caveats=[], max_rejection_reasons=3,
    )
    assert len(payload.rejection_reasons) == 3


def test_no_wmape_and_not_fallback_still_produces_a_summary():
    payload = build_template_insight(
        selected_model="prophet", wmape=None, is_fallback=False, fallback_trigger=None,
        rejected_candidates=[], confidence_pct=None, caveats=[],
    )
    assert payload.concise_summary
    assert "prophet" in payload.concise_summary.lower()


def test_summary_stays_within_the_ui_length_cap():
    from forecast_engine.s11_llm.schema import MAX_SUMMARY_WORDS

    payload = build_template_insight(
        selected_model="xgboost", wmape=25.86, is_fallback=False, fallback_trigger=None,
        rejected_candidates=[], confidence_pct=74.1, caveats=[],
    )
    assert len(payload.concise_summary.split()) <= MAX_SUMMARY_WORDS


def test_the_template_path_always_passes_schema_validation():
    from forecast_engine.s11_llm.schema import parse_and_validate
    import json

    payload = build_template_insight(
        selected_model="xgboost", wmape=25.86, is_fallback=False, fallback_trigger=None,
        rejected_candidates=[{"model_name": "prophet", "reason": "eliminated"}],
        confidence_pct=74.1, caveats=[],
    )
    # Round-trips through the exact validator the LLM path uses.
    revalidated = parse_and_validate(json.dumps(payload.to_dict()), expected_model="xgboost")
    assert revalidated.selected_model == "xgboost"
