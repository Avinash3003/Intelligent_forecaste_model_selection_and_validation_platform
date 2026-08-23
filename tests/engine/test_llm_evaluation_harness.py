"""The held-out regression evaluation mechanism (Section 13.3)."""

from forecast_engine.s11_llm.evaluation import default_regression_set, run_evaluation
from forecast_engine.s11_llm.schema import InsightPayload
from forecast_engine.s11_llm.template_fallback import build_template_insight


def _template_generate(case):
    return build_template_insight(
        selected_model=case.selected_model, wmape=case.wmape, is_fallback=case.is_fallback,
        fallback_trigger=case.fallback_trigger, rejected_candidates=case.rejected_candidates,
        confidence_pct=case.confidence_pct, caveats=case.caveats,
    )


def test_the_default_regression_set_is_non_trivial():
    cases = default_regression_set()
    assert len(cases) >= 5
    assert {case.is_fallback for case in cases} == {True, False}


def test_the_deterministic_template_path_scores_perfectly_grounded():
    # The template fallback only ever writes numbers straight from its
    # inputs, so it is the harness's own sanity check: if this doesn't
    # score 100% grounded, the grounding checker itself has regressed.
    report = run_evaluation(default_regression_set(), _template_generate)
    assert report.groundedness_rate == 1.0
    assert report.hallucination_rate == 0.0


def test_every_case_stays_within_the_length_cap():
    report = run_evaluation(default_regression_set(), _template_generate)
    assert report.length_compliance_rate == 1.0


def test_every_case_produces_a_payload():
    report = run_evaluation(default_regression_set(), _template_generate)
    assert report.generated_count == report.case_count


def test_a_generator_that_hallucinates_is_caught():
    def bad_generate(case):
        return InsightPayload(
            selected_model=case.selected_model,
            concise_summary=f"{case.selected_model} achieved a WMAPE of 999.9%, a record result.",
        )

    report = run_evaluation(default_regression_set(), bad_generate)
    assert report.groundedness_rate is not None
    assert report.groundedness_rate < 1.0
    assert report.hallucination_rate > 0.0


def test_a_generator_that_raises_is_recorded_not_fatal():
    def broken_generate(case):
        raise RuntimeError("simulated provider outage")

    report = run_evaluation(default_regression_set(), broken_generate)
    assert report.generated_count == 0
    assert report.case_count == len(default_regression_set())
    assert all(r.generation_error for r in report.results)


def test_a_generator_that_writes_an_essay_fails_length_compliance():
    def verbose_generate(case):
        return InsightPayload(
            selected_model=case.selected_model,
            concise_summary=" ".join(["word"] * 200),
        )

    report = run_evaluation(default_regression_set(), verbose_generate)
    assert report.length_compliance_rate == 0.0


def test_report_serializes_to_a_plain_dict():
    report = run_evaluation(default_regression_set(), _template_generate)
    payload = report.to_dict()
    assert payload["case_count"] == len(default_regression_set())
    assert isinstance(payload["results"], list)
