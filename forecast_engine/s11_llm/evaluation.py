"""LLM evaluation harness (Section 13.3, "Evaluation").

Section 13.3 asks for a held-out regression set of key/model/rejection-
reason combinations, run whenever the prompt template or underlying model
changes, tracking groundedness, hallucination/unsupported-claim rate, and
narrative length/readability — "not simply whether an explanation was
generated successfully."

This harness is generation-source-agnostic: `run_evaluation` takes a
`generate` callable and scores whatever `InsightPayload` it returns against
each case's known-correct facts. Pointed at `template_fallback.
build_template_insight`, it runs with no network access and no credentials
— a real, CI-safe regression suite that catches a grounding-checker
regression or a schema change today. Pointed at a live
`LLMInsightEngine`-style call, the exact same harness becomes the "did this
prompt/model change hurt quality" suite Section 13.3 describes; that path
just is not exercised in this environment, which has no Azure OpenAI
credentials to call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from forecast_engine.s11_llm.grounding import check_grounding
from forecast_engine.s11_llm.schema import MAX_SUMMARY_WORDS, InsightPayload


@dataclass
class EvalCase:
    """One held-out (key, model, rejection-reason) combination with known
    facts — the ground truth a generated insight is scored against."""

    case_id: str
    selected_model: str
    wmape: float | None
    is_fallback: bool
    fallback_trigger: str | None
    rejected_candidates: list[dict[str, str]] = field(default_factory=list)
    confidence_pct: float | None = None
    caveats: list[str] = field(default_factory=list)
    # Extra grounding facts beyond wmape/confidence (e.g. a drift
    # statistic) a generator might legitimately cite.
    extra_metrics: dict[str, float | None] = field(default_factory=dict)

    def metrics(self) -> dict[str, float | None]:
        return {"wmape": self.wmape, "confidence_estimate": self.confidence_pct, **self.extra_metrics}


@dataclass
class EvalCaseResult:
    case_id: str
    payload: InsightPayload | None
    grounded: bool
    grounding_issues: list[str]
    word_count: int
    within_length_cap: bool
    generation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "insight": self.payload.to_dict() if self.payload else None,
            "grounded": self.grounded,
            "grounding_issues": self.grounding_issues,
            "word_count": self.word_count,
            "within_length_cap": self.within_length_cap,
            "generation_error": self.generation_error,
        }


@dataclass
class EvalReport:
    results: list[EvalCaseResult] = field(default_factory=list)

    @property
    def case_count(self) -> int:
        return len(self.results)

    @property
    def generated_count(self) -> int:
        return sum(1 for r in self.results if r.payload is not None)

    @property
    def groundedness_rate(self) -> float | None:
        graded = [r for r in self.results if r.payload is not None]
        if not graded:
            return None
        return sum(1 for r in graded if r.grounded) / len(graded)

    @property
    def hallucination_rate(self) -> float | None:
        # The complement of groundedness, reported separately because
        # Section 13.3 names it as its own tracked metric — a reader
        # looking for "how often did it make something up" should not have
        # to compute 1 - groundedness themselves.
        rate = self.groundedness_rate
        return None if rate is None else round(1.0 - rate, 4)

    @property
    def length_compliance_rate(self) -> float | None:
        graded = [r for r in self.results if r.payload is not None]
        if not graded:
            return None
        return sum(1 for r in graded if r.within_length_cap) / len(graded)

    @property
    def average_word_count(self) -> float | None:
        graded = [r.word_count for r in self.results if r.payload is not None]
        if not graded:
            return None
        return sum(graded) / len(graded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_count": self.case_count,
            "generated_count": self.generated_count,
            "groundedness_rate": (
                round(self.groundedness_rate, 4) if self.groundedness_rate is not None else None
            ),
            "hallucination_rate": self.hallucination_rate,
            "length_compliance_rate": (
                round(self.length_compliance_rate, 4) if self.length_compliance_rate is not None else None
            ),
            "average_word_count": (
                round(self.average_word_count, 1) if self.average_word_count is not None else None
            ),
            "results": [r.to_dict() for r in self.results],
        }


def run_evaluation(
    cases: list[EvalCase], generate: Callable[[EvalCase], InsightPayload]
) -> EvalReport:
    """Run every case through `generate` and score the result.

    `generate` never needs to raise for a case to be scored honestly — a
    raised exception is caught and recorded as `generation_error`, and that
    case contributes to `case_count` but not `generated_count`, so a
    generator that fails outright shows up as a coverage gap rather than
    disappearing from the report.
    """
    results: list[EvalCaseResult] = []
    for case in cases:
        try:
            payload = generate(case)
        except Exception as exc:  # noqa: BLE001 - a bad case must not stop the suite
            results.append(
                EvalCaseResult(
                    case_id=case.case_id, payload=None, grounded=False, grounding_issues=[],
                    word_count=0, within_length_cap=False, generation_error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        grounding = check_grounding(payload.concise_summary, case.metrics())
        word_count = len(payload.concise_summary.split())
        results.append(
            EvalCaseResult(
                case_id=case.case_id,
                payload=payload,
                grounded=grounding.grounded,
                grounding_issues=grounding.issues,
                word_count=word_count,
                within_length_cap=word_count <= MAX_SUMMARY_WORDS,
            )
        )
    return EvalReport(results=results)


def default_regression_set() -> list[EvalCase]:
    """A small held-out set spanning the decision shapes the platform
    actually produces — a clean win, a multi-rejection win, a fallback
    from drift, a fallback from full elimination, thin evidence, and a
    near-zero WMAPE edge case. Not exhaustive; enough to catch a regression
    in the grounding checker, the schema, or the template generator.
    """
    return [
        EvalCase(
            case_id="clean_win",
            selected_model="xgboost",
            wmape=8.2,
            is_fallback=False,
            fallback_trigger=None,
            rejected_candidates=[],
            confidence_pct=91.8,
        ),
        EvalCase(
            case_id="multi_rejection_win",
            selected_model="lightgbm",
            wmape=15.5,
            is_fallback=False,
            fallback_trigger=None,
            rejected_candidates=[
                {"model_name": "prophet", "reason": "Eliminated at forward validation: missing seasonality"},
                {"model_name": "arima", "reason": "Rejected by ranking: lower composite score"},
            ],
            confidence_pct=78.3,
        ),
        EvalCase(
            case_id="fallback_from_drift",
            selected_model="seasonal_naive",
            wmape=None,
            is_fallback=True,
            fallback_trigger="All 1 evaluated model(s) failed validation (xgboost).",
            rejected_candidates=[
                {"model_name": "xgboost", "reason": "wasserstein_distance statistic 0.89 > threshold 0.73."}
            ],
            confidence_pct=94.0,
        ),
        EvalCase(
            case_id="fallback_full_elimination",
            selected_model="seasonal_naive",
            wmape=None,
            is_fallback=True,
            fallback_trigger="No ranked candidate survived earlier validation stages.",
            rejected_candidates=[],
            confidence_pct=None,
        ),
        EvalCase(
            case_id="thin_evidence_short_history",
            selected_model="arima",
            wmape=22.4,
            is_fallback=False,
            fallback_trigger=None,
            rejected_candidates=[{"model_name": "tft", "reason": "Skipped: insufficient history (24 < 60)."}],
            confidence_pct=61.0,
            caveats=["short history"],
        ),
        EvalCase(
            case_id="near_perfect_backtest",
            selected_model="prophet",
            wmape=0.4,
            is_fallback=False,
            fallback_trigger=None,
            rejected_candidates=[],
            confidence_pct=99.6,
        ),
    ]
