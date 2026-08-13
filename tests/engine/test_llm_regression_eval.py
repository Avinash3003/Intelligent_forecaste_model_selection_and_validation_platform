"""The LLM Evaluation + Regression Framework (Section 13.3, extended).

Every check here is exercised against the deterministic template path
(`build_template_insight`) for the "should pass" cases, and against small,
hand-built `InsightPayload`s for the "should catch this" cases — no network,
no credentials, CI-safe, mirroring the existing `test_llm_evaluation_harness.py`
pattern this module extends rather than replaces.
"""

from __future__ import annotations

import json

import pytest

from forecast_engine.s11_llm.evaluation import EvalCase
from forecast_engine.s11_llm.regression_eval import (
    CheckResult,
    RegressionReport,
    RegressionThresholds,
    check_groundedness,
    check_readability,
    check_rejection_accuracy,
    check_schema_validity,
    check_winner_consistency,
    evaluate_case,
    evaluate_regression_set,
    load_eval_dataset,
)
from forecast_engine.s11_llm.schema import InsightPayload
from forecast_engine.s11_llm.template_fallback import build_template_insight


def _case(**overrides) -> EvalCase:
    defaults = dict(
        case_id="case-1",
        selected_model="xgboost",
        wmape=10.0,
        is_fallback=False,
        fallback_trigger=None,
        rejected_candidates=[{"model_name": "prophet", "reason": "Eliminated at forward validation: excessive smoothing"}],
        confidence_pct=85.0,
        caveats=[],
    )
    defaults.update(overrides)
    return EvalCase(**defaults)


def _template_generate(case: EvalCase):
    payload = build_template_insight(
        selected_model=case.selected_model, wmape=case.wmape, is_fallback=case.is_fallback,
        fallback_trigger=case.fallback_trigger, rejected_candidates=case.rejected_candidates,
        confidence_pct=case.confidence_pct, caveats=case.caveats,
    )
    return payload, None


# ---------------------------------------------------------------------
# Dataset fixture
# ---------------------------------------------------------------------


def test_the_bundled_dataset_loads_and_meets_the_size_target():
    dataset_version, cases = load_eval_dataset()
    assert dataset_version
    assert 10 <= len(cases) <= 20
    assert len(cases) == len({c.case_id for c in cases})  # unique ids


def test_the_bundled_dataset_covers_every_required_scenario():
    _, cases = load_eval_dataset()
    scenarios = {c.scenario for c in cases}
    required = {
        "normal_winner", "multi_rejection", "fallback_winner", "strong_accuracy",
        "weak_accuracy", "excessive_smoothing", "missing_seasonality", "missing_trend",
        "drift_warning",
    }
    assert required <= scenarios


def test_every_bundled_case_scores_perfectly_through_the_template_path():
    # The deterministic path is grounded and schema-valid by construction —
    # the harness's own sanity check, same rationale as evaluation.py's.
    _, cases = load_eval_dataset()
    report = evaluate_regression_set(cases, _template_generate, dataset_version="v1", generation_mode="template")
    assert report.overall_pass_rate == 1.0
    assert report.regression_passed


# ---------------------------------------------------------------------
# Schema validity
# ---------------------------------------------------------------------


def test_schema_validity_passes_a_well_formed_raw_response():
    case = _case(rejected_candidates=[])
    raw = json.dumps({
        "selected_model": "xgboost", "rejection_reasons": [], "confidence": 85.0,
        "caveats": [], "concise_summary": "xgboost was selected.",
    })
    result = check_schema_validity(case, InsightPayload(selected_model="xgboost"), raw)
    assert result.passed


def test_schema_validity_fails_malformed_json():
    case = _case()
    result = check_schema_validity(case, None, "not json at all")
    assert not result.passed


def test_schema_validity_reuses_expected_model_mismatch_detection():
    case = _case(selected_model="xgboost")
    raw = json.dumps({
        "selected_model": "prophet", "rejection_reasons": [], "confidence": 85.0,
        "caveats": [], "concise_summary": "prophet was selected.",
    })
    result = check_schema_validity(case, None, raw)
    assert not result.passed


def test_schema_validity_treats_a_template_built_payload_as_valid_without_raw_text():
    payload = build_template_insight(
        selected_model="xgboost", wmape=10.0, is_fallback=False, fallback_trigger=None,
        rejected_candidates=[], confidence_pct=85.0, caveats=[],
    )
    result = check_schema_validity(_case(), payload, None)
    assert result.passed


# ---------------------------------------------------------------------
# Winner consistency
# ---------------------------------------------------------------------


def test_winner_consistency_passes_on_exact_match():
    case = _case(selected_model="xgboost")
    result = check_winner_consistency(case, InsightPayload(selected_model="xgboost"))
    assert result.passed


def test_winner_consistency_is_case_insensitive():
    case = _case(selected_model="XGBoost")
    result = check_winner_consistency(case, InsightPayload(selected_model="xgboost"))
    assert result.passed


def test_winner_consistency_fails_when_the_llm_names_a_rejected_model():
    case = _case(selected_model="xgboost", rejected_candidates=[{"model_name": "prophet", "reason": "x"}])
    result = check_winner_consistency(case, InsightPayload(selected_model="prophet"))
    assert not result.passed
    assert "prophet" in result.detail.lower()


# ---------------------------------------------------------------------
# Rejection-reason accuracy
# ---------------------------------------------------------------------


def test_rejection_accuracy_passes_when_no_candidates_were_rejected():
    case = _case(rejected_candidates=[])
    result = check_rejection_accuracy(case, InsightPayload(selected_model="xgboost"))
    assert result.passed


def test_rejection_accuracy_passes_on_a_category_match_not_exact_wording():
    case = _case(rejected_candidates=[
        {"model_name": "prophet", "reason": "Eliminated at forward validation: excessive smoothing"}
    ])
    payload = InsightPayload(
        selected_model="xgboost",
        rejection_reasons=["overly smooth forecast"],
        concise_summary="xgboost was selected. It outperformed prophet, which had an overly smooth forecast.",
    )
    result = check_rejection_accuracy(case, payload)
    assert result.passed


def test_rejection_accuracy_passes_when_the_model_name_is_only_in_the_summary_not_the_reason_list():
    # The real-world shape confirmed against Azure OpenAI: rejection_reasons
    # holds the bare category phrase (matching the v2 schema's own "one
    # short phrase per rejected candidate", no model-name field), while the
    # candidate's name appears in the narrative sentence introducing it.
    case = _case(rejected_candidates=[
        {"model_name": "prophet", "reason": "Eliminated at forward validation: missing seasonality"}
    ])
    payload = InsightPayload(
        selected_model="xgboost",
        rejection_reasons=["missing seasonality"],
        concise_summary="xgboost was selected. It outperformed prophet, which was eliminated for missing seasonality.",
    )
    result = check_rejection_accuracy(case, payload)
    assert result.passed


def test_rejection_accuracy_fails_when_a_rejected_model_is_never_mentioned():
    case = _case(rejected_candidates=[{"model_name": "prophet", "reason": "excessive smoothing"}])
    payload = InsightPayload(selected_model="xgboost", rejection_reasons=[], concise_summary="xgboost was selected.")
    result = check_rejection_accuracy(case, payload)
    assert not result.passed
    assert "prophet" in result.detail.lower()


def test_rejection_accuracy_fails_on_a_wrong_category_even_if_the_model_is_named():
    case = _case(rejected_candidates=[{"model_name": "prophet", "reason": "excessive smoothing"}])
    payload = InsightPayload(
        selected_model="xgboost", rejection_reasons=["lower composite score"],
        concise_summary="xgboost was selected. prophet was rejected for a lower composite score.",
    )
    result = check_rejection_accuracy(case, payload)
    assert not result.passed


def test_rejection_accuracy_does_not_let_one_candidates_category_cover_another():
    # Regression guard for the "one bag of words over the whole response"
    # version of this check, which would have wrongly passed this.
    case = _case(rejected_candidates=[
        {"model_name": "prophet", "reason": "excessive smoothing"},
        {"model_name": "arima", "reason": "missing trend"},
    ])
    payload = InsightPayload(
        selected_model="xgboost", rejection_reasons=["excessive smoothing"],
        concise_summary="xgboost was selected. prophet had an overly smooth forecast.",
    )
    result = check_rejection_accuracy(case, payload)
    assert not result.passed
    assert "arima" in result.detail.lower()


# ---------------------------------------------------------------------
# Groundedness / hallucination taxonomy
# ---------------------------------------------------------------------


def test_groundedness_passes_a_number_that_matches_context():
    case = _case(wmape=25.98, rejected_candidates=[])
    payload = InsightPayload(selected_model="xgboost", concise_summary="xgboost achieved 25.98% WMAPE.")
    result, category = check_groundedness(case, payload)
    assert result.passed
    assert category == "grounded"


def test_groundedness_fails_an_unsupported_number():
    case = _case(wmape=25.98, rejected_candidates=[])
    payload = InsightPayload(selected_model="xgboost", concise_summary="xgboost achieved 18.2% WMAPE.")
    result, category = check_groundedness(case, payload)
    assert not result.passed
    assert category == "unsupported"


def test_groundedness_categorizes_a_wrong_winner_as_contradictory_not_merely_unsupported():
    case = _case(selected_model="xgboost", rejected_candidates=[])
    payload = InsightPayload(selected_model="prophet", concise_summary="prophet was selected.")
    result, category = check_groundedness(case, payload)
    assert not result.passed
    assert category == "contradictory"


# ---------------------------------------------------------------------
# Readability
# ---------------------------------------------------------------------


def test_readability_passes_a_normal_summary():
    case = _case(rejected_candidates=[])
    payload = InsightPayload(selected_model="xgboost", concise_summary="xgboost was selected, with a backtest WMAPE of 10.0%.")
    assert check_readability(case, payload).passed


def test_readability_fails_when_the_winner_is_not_named_first():
    case = _case(selected_model="xgboost", rejected_candidates=[])
    payload = InsightPayload(selected_model="xgboost", concise_summary="The pipeline evaluated several candidates before choosing xgboost.")
    result = check_readability(case, payload)
    assert not result.passed
    assert "first sentence" in result.detail


def test_readability_fails_when_it_opens_with_a_rejected_candidate():
    case = _case(selected_model="xgboost", rejected_candidates=[{"model_name": "prophet", "reason": "x"}])
    payload = InsightPayload(selected_model="xgboost", concise_summary="Prophet was rejected, so xgboost was selected instead.")
    result = check_readability(case, payload)
    assert not result.passed


def test_readability_fails_on_leaked_json():
    case = _case(rejected_candidates=[])
    payload = InsightPayload(selected_model="xgboost", concise_summary='xgboost was selected. {"confidence": 85}')
    assert not check_readability(case, payload).passed


def test_readability_fails_on_verbatim_repetition():
    case = _case(rejected_candidates=[])
    text = "xgboost was selected with strong accuracy. xgboost was selected with strong accuracy."
    payload = InsightPayload(selected_model="xgboost", concise_summary=text)
    assert not check_readability(case, payload).passed


def test_readability_fails_over_the_word_cap():
    case = _case(rejected_candidates=[])
    payload = InsightPayload(selected_model="xgboost", concise_summary="xgboost was selected. " + " ".join(["word"] * 80))
    result = check_readability(case, payload)
    assert not result.passed
    assert "words exceeds" in result.detail


def test_readability_fails_on_an_empty_summary():
    case = _case(rejected_candidates=[])
    payload = InsightPayload(selected_model="xgboost", concise_summary="")
    assert not check_readability(case, payload).passed


# ---------------------------------------------------------------------
# Per-case orchestration
# ---------------------------------------------------------------------


def test_evaluate_case_records_a_hard_generation_failure_without_crashing():
    def broken_generate(case):
        raise RuntimeError("simulated provider outage")

    result = evaluate_case(_case(), broken_generate)
    assert result.generation_error is not None
    assert not result.overall_pass
    assert result.hallucination_category == "not_generated"


def test_evaluate_case_still_scores_a_schema_invalid_response_not_just_skips_it():
    def bad_generate(case):
        return None, "not valid json"

    result = evaluate_case(_case(), bad_generate)
    assert result.generation_error is None  # a real (bad) response, not a crash
    checks = {c.name: c.passed for c in result.checks}
    assert checks["schema_validity"] is False
    assert not result.overall_pass


def test_evaluate_case_passes_every_check_for_the_deterministic_path():
    result = evaluate_case(_case(rejected_candidates=[]), _template_generate)
    assert result.overall_pass
    assert result.hallucination_category == "grounded"


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


def test_aggregate_rates_are_computed_correctly():
    cases = [_case(case_id=f"c{i}", rejected_candidates=[]) for i in range(4)]

    def flaky_generate(case):
        if case.case_id == "c3":
            return InsightPayload(selected_model="wrong-model", concise_summary="wrong-model was selected."), None
        return _template_generate(case)

    report = evaluate_regression_set(cases, flaky_generate, dataset_version="v1", generation_mode="test")
    assert report.case_count == 4
    assert report.generated_count == 4  # no hard failures, just one bad response
    assert report.winner_consistency_rate == 0.75
    assert report.overall_pass_rate == 0.75


def test_hard_generation_failures_are_excluded_from_rate_denominators_but_counted_as_violations():
    cases = [_case(case_id="c1", rejected_candidates=[]), _case(case_id="c2", rejected_candidates=[])]

    def one_broken(case):
        if case.case_id == "c2":
            raise RuntimeError("boom")
        return _template_generate(case)

    report = evaluate_regression_set(cases, one_broken, dataset_version="v1", generation_mode="test")
    assert report.case_count == 2
    assert report.generated_count == 1
    assert report.groundedness_rate == 1.0  # only the one real response is graded
    violations = report.threshold_violations()
    assert any("failed to generate" in v for v in violations)


def test_report_serializes_to_a_plain_dict_with_every_field():
    _, cases = load_eval_dataset()
    report = evaluate_regression_set(cases[:3], _template_generate, dataset_version="v1", prompt_version="v2", generation_mode="template")
    payload = report.to_dict()
    for key in (
        "dataset_version", "prompt_version", "generation_mode", "case_count", "schema_pass_rate",
        "groundedness_rate", "winner_consistency_rate", "rejection_accuracy_rate", "hallucination_rate",
        "readability_pass_rate", "overall_pass_rate", "thresholds", "regression_passed",
        "threshold_violations", "results",
    ):
        assert key in payload
    assert len(payload["results"]) == 3


# ---------------------------------------------------------------------
# Thresholds / PASS-FAIL
# ---------------------------------------------------------------------


def test_default_thresholds_are_practical_for_a_small_set():
    thresholds = RegressionThresholds.default()
    # A single miss out of 16 cases (~6.25%) must not fail every metric —
    # that would make the suite unusable at its documented target size.
    assert thresholds.minimum_groundedness <= 0.9375
    assert thresholds.minimum_rejection_accuracy <= 0.9375
    # Winner consistency is the one zero-tolerance gate.
    assert thresholds.minimum_winner_consistency == 1.0


def test_thresholds_from_dict_ignores_unknown_keys_and_fills_defaults():
    thresholds = RegressionThresholds.from_dict({"minimum_groundedness": 0.5, "not_a_real_field": 123})
    assert thresholds.minimum_groundedness == 0.5
    assert thresholds.minimum_schema_pass_rate == RegressionThresholds.default().minimum_schema_pass_rate


def test_a_report_meeting_every_threshold_passes():
    _, cases = load_eval_dataset()
    report = evaluate_regression_set(cases, _template_generate, dataset_version="v1", generation_mode="template")
    assert report.regression_passed
    assert report.threshold_violations() == []


def test_a_report_violating_a_threshold_fails_with_a_specific_reason():
    cases = [_case(case_id=f"c{i}", selected_model="xgboost", rejected_candidates=[]) for i in range(4)]

    def always_wrong(case):
        return InsightPayload(selected_model="not-the-winner", concise_summary="not-the-winner was selected."), None

    report = evaluate_regression_set(
        cases, always_wrong, dataset_version="v1", generation_mode="test",
        thresholds=RegressionThresholds(minimum_winner_consistency=1.0),
    )
    assert not report.regression_passed
    assert any("winner_consistency_rate" in v for v in report.threshold_violations())


def test_thresholds_ignore_a_metric_with_no_graded_cases_rather_than_dividing_by_zero():
    report = evaluate_regression_set([], _template_generate, dataset_version="v1", generation_mode="template")
    assert report.groundedness_rate is None
    assert report.case_count == 0
    # No graded cases and zero total cases is not itself a violation — an
    # empty run is a configuration question, not a quality regression.
    assert report.threshold_violations() == []
