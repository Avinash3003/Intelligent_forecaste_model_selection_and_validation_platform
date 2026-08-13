"""CLI entry point for the LLM Evaluation + Regression Framework.

    python -m forecast_engine.s11_llm.evaluate

Runs the version-controlled evaluation dataset (`eval_dataset/
regression_cases.json`) through the currently configured prompt version and
model — real Azure OpenAI when credentials are configured, the
deterministic template path otherwise — scores every case, applies the
configured regression thresholds, and writes a JSON report other tooling
(the backend's Observability API) can read. Exits 0 on a passing
regression, 1 on a failing one, 2 if `--mode llm` was requested but Azure
OpenAI is not actually configured.

Nothing here touches the production pipeline: it never runs
`forecast_engine.run_pipeline`, never opens an MLflow run, and never writes
anywhere `run_pipeline` itself reads from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forecast_engine.config.llm_config import LLMConfig
from forecast_engine.s11_llm.azure_openai_service import AzureOpenAIService
from forecast_engine.s11_llm.evaluation import EvalCase
from forecast_engine.s11_llm.prompt_library import STRUCTURED_INSIGHT_TEMPLATE, PromptLibrary
from forecast_engine.s11_llm.regression_eval import (
    GenerateFn,
    RegressionReport,
    RegressionThresholds,
    evaluate_case,
    load_eval_dataset,
)
from forecast_engine.s11_llm.schema import InsightPayload, SchemaValidationError, parse_and_validate
from forecast_engine.s11_llm.template_fallback import build_template_insight

_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "eval_output" / "latest_regression_report.json"

_CHECK_LABELS = {
    "schema_validity": "Schema",
    "groundedness": "Groundedness",
    "winner_consistency": "Winner consistency",
    "rejection_accuracy": "Rejection accuracy",
    "readability": "Readability",
}


def _template_generate(case: EvalCase) -> tuple[InsightPayload, None]:
    """The deterministic, no-network path — perfectly grounded by
    construction, and the harness's own sanity check (see
    `evaluation.py`'s module docstring)."""
    payload = build_template_insight(
        selected_model=case.selected_model,
        wmape=case.wmape,
        is_fallback=case.is_fallback,
        fallback_trigger=case.fallback_trigger,
        rejected_candidates=case.rejected_candidates,
        confidence_pct=case.confidence_pct,
        caveats=case.caveats,
    )
    return payload, None


def _case_context(case: EvalCase) -> str:
    """A minimal decision-record context block for one eval case, in the
    same style `context_formatter.render_group_context` uses for a real
    pipeline run — built directly from the case's own known facts, since
    an eval case has no real `PipelineResult` behind it to render from.
    """
    lines = [
        "## Final Production Model Selection (Section 6.9)",
        f"model={case.selected_model}, fallback={case.is_fallback}"
        + (f", wmape={case.wmape}" if case.wmape is not None else "")
        + (f", confidence={case.confidence_pct}" if case.confidence_pct is not None else ""),
    ]
    if case.rejected_candidates:
        lines.append(
            "Rejected candidates: "
            + str([(c.get("model_name"), c.get("reason")) for c in case.rejected_candidates])
        )
    if case.is_fallback and case.fallback_trigger:
        lines.append(f"Fallback trigger: {case.fallback_trigger}")
    if case.caveats:
        lines.append(f"Caveats: {', '.join(case.caveats)}")
    if case.extra_metrics:
        lines.append("## Drift Detection & Threshold Estimation (Sections 6.7-6.9)")
        lines.append(", ".join(f"{k}={v}" for k, v in case.extra_metrics.items()))
    return "\n".join(lines)


def _make_llm_generate(prompt_version: str) -> tuple[LLMConfig, AzureOpenAIService, GenerateFn]:
    config = LLMConfig.default()
    if prompt_version != config.prompt_version:
        config.prompt_version = prompt_version  # explicit CLI override wins
    service = AzureOpenAIService(config)
    prompts = PromptLibrary(version=prompt_version)

    def _generate(case: EvalCase) -> tuple[InsightPayload | None, str | None]:
        context = _case_context(case)
        user_prompt = prompts.render(
            STRUCTURED_INSIGHT_TEMPLATE,
            {"context": context, "max_rejections": max(len(case.rejected_candidates), 1)},
        )
        system_prompt = prompts.system_prompt()
        result = service.complete(system_prompt, user_prompt, json_mode=True)
        try:
            payload = parse_and_validate(result.text, expected_model=case.selected_model)
        except SchemaValidationError:
            # Still return the raw text — evaluate_case()'s own schema
            # check re-derives this exact failure from it, so the case is
            # graded as a schema failure, not silently dropped.
            return None, result.text
        return payload, result.text

    return config, service, _generate


def _print_report(report: RegressionReport) -> None:
    for r in report.results:
        header = f"Case: {r.case_id}"
        if r.scenario:
            header += f" [{r.scenario}]"
        print(f"\n{header}")
        if r.generation_error:
            print(f"  Generation error: {r.generation_error}")
            print("  Overall: FAIL")
            continue
        for check in r.checks:
            label = _CHECK_LABELS.get(check.name, check.name)
            line = f"  {label + ':':<22}{'PASS' if check.passed else 'FAIL'}"
            if not check.passed and check.detail:
                line += f"  ({check.detail})"
            print(line)
        hallucination_ok = r.hallucination_category == "grounded"
        print(
            f"  {'Hallucination:':<22}{'PASS' if hallucination_ok else 'FAIL'}"
            + ("" if hallucination_ok else f"  ({r.hallucination_category})")
        )
        print(f"  Overall: {'PASS' if r.overall_pass else 'FAIL'}")

    def pct(value: float | None) -> str:
        return "—" if value is None else f"{value:.1%}"

    print("\n" + "=" * 56)
    print(f"Evaluation dataset:     {report.dataset_version}")
    print(f"Prompt version:         {report.prompt_version}")
    print(f"Generation:             {report.generation_mode}")
    print(f"Cases evaluated:        {report.case_count}")
    print(f"Schema pass rate:       {pct(report.schema_pass_rate)}")
    print(f"Groundedness:           {pct(report.groundedness_rate)}")
    print(f"Winner consistency:     {pct(report.winner_consistency_rate)}")
    print(f"Rejection accuracy:     {pct(report.rejection_accuracy_rate)}")
    print(f"Hallucination rate:     {pct(report.hallucination_rate)}")
    print(f"Readability pass rate:  {pct(report.readability_pass_rate)}")
    print(f"Overall pass rate:      {pct(report.overall_pass_rate)}")

    violations = report.threshold_violations()
    print("\nRegression result: " + ("PASS" if not violations else "FAIL"))
    for v in violations:
        print(f"  - {v}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forecast_engine.s11_llm.evaluate",
        description="Run the LLM evaluation/regression suite against the evaluation dataset.",
    )
    parser.add_argument("--dataset", default=None, help="Path to the eval dataset JSON (default: bundled fixture).")
    parser.add_argument("--prompt-version", default="v2", help="Prompt version to evaluate (default: v2).")
    parser.add_argument(
        "--mode",
        choices=["auto", "llm", "template"],
        default="auto",
        help="'llm' calls Azure OpenAI and fails if it is not configured; 'template' uses only the "
        "deterministic, no-network fallback path; 'auto' (default) uses the LLM when configured, "
        "template otherwise.",
    )
    parser.add_argument("--thresholds", default=None, help="Path to a JSON file overriding default thresholds.")
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT), help="Where to write the JSON report.")
    args = parser.parse_args(argv)

    dataset_version, cases = load_eval_dataset(args.dataset)

    thresholds = RegressionThresholds.default()
    if args.thresholds:
        thresholds = RegressionThresholds.from_dict(json.loads(Path(args.thresholds).read_text()))

    generation_mode = "template"
    generate: GenerateFn = _template_generate

    if args.mode in ("llm", "auto"):
        config, service, llm_generate = _make_llm_generate(args.prompt_version)
        if service.is_available():
            generate = llm_generate
            generation_mode = f"azure_openai:{config.deployment_name}"
        elif args.mode == "llm":
            print(f"Azure OpenAI is not available: {service.unavailable_reason()}", file=sys.stderr)
            return 2
        else:
            print(
                f"Azure OpenAI is not available ({service.unavailable_reason()}); "
                "falling back to the deterministic template path.",
                file=sys.stderr,
            )

    results = [evaluate_case(case, generate) for case in cases]
    report = RegressionReport(
        dataset_version=dataset_version,
        prompt_version=args.prompt_version,
        generation_mode=generation_mode,
        results=results,
        thresholds=thresholds,
    )

    _print_report(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote report to {output_path}")

    return 0 if report.regression_passed else 1


if __name__ == "__main__":
    sys.exit(main())
