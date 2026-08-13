"""LLM Evaluation + Regression Framework (Section 13.3) — the "when we
change the prompt or model, is the explanation still correct" layer.

Deliberately additive, not a replacement: `evaluation.py`'s `run_evaluation`
already scores groundedness (via `grounding.check_grounding`, unmodified)
and length-cap compliance correctly, and nothing here recomputes those —
`check_groundedness` below calls the exact same `check_grounding` function.
What this module adds is everything `run_evaluation` does not attempt:
winner consistency, rejection-reason accuracy (against the engine's own
rejection reasons, by category rather than exact wording), an explicit
hallucination taxonomy (grounded / unsupported / contradictory, not just a
grounded/not boolean), readability rules the v2 prompt itself specifies
(winner named first, no repetition, no leaked JSON), schema validity (by
reusing `schema.parse_and_validate` when the raw model response is
available), and configurable pass/fail regression thresholds over the
whole set.

Every check is deterministic — string/keyword comparison against the
structured ground truth each `EvalCase` already carries, never a second LLM
call grading the first one. That keeps the suite free, fast, and reusable
in CI without new credentials.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from forecast_engine.s11_llm.evaluation import EvalCase
from forecast_engine.s11_llm.grounding import check_grounding
from forecast_engine.s11_llm.schema import (
    MAX_SUMMARY_WORDS,
    InsightPayload,
    SchemaValidationError,
    parse_and_validate,
)

_DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset" / "regression_cases.json"

# A case's `generate` callable returns (payload, raw_text): `payload` is the
# parsed/validated InsightPayload (None if generation or validation failed);
# `raw_text` is the model's own raw response when one exists (None for the
# deterministic template path, which never produces "raw" text to validate
# a second time) — carried through so `check_schema_validity` can reuse the
# real validator instead of re-deriving pass/fail from the already-parsed
# payload.
GenerateFn = Callable[[EvalCase], tuple[InsightPayload | None, str | None]]


def load_eval_dataset(path: str | Path | None = None) -> tuple[str, list[EvalCase]]:
    """Load the version-controlled evaluation dataset.

    Returns `(dataset_version, cases)`. The dataset is a small JSON fixture
    (`eval_dataset/regression_cases.json`) rather than inline Python so it
    reviews as a diff of data, not code, and so a non-engineer can extend
    coverage without touching this module.
    """
    resolved = Path(path) if path else _DEFAULT_DATASET_PATH
    payload = json.loads(resolved.read_text())
    cases = [
        EvalCase(
            case_id=c["case_id"],
            scenario=c.get("scenario"),
            selected_model=c["selected_model"],
            wmape=c.get("wmape"),
            is_fallback=bool(c.get("is_fallback")),
            fallback_trigger=c.get("fallback_trigger"),
            rejected_candidates=list(c.get("rejected_candidates") or []),
            confidence_pct=c.get("confidence_pct"),
            caveats=list(c.get("caveats") or []),
            extra_metrics=dict(c.get("extra_metrics") or {}),
        )
        for c in payload.get("cases", [])
    ]
    return str(payload.get("dataset_version", "unknown")), cases


# ---------------------------------------------------------------------
# Rejection-reason categories — deterministic keyword matching, not NLP.
# ---------------------------------------------------------------------
#
# The goal is "does the LLM's stated reason belong to the same category as
# the engine's real reason", not exact wording — Section 13.3 explicitly
# does not require that. A category is a small, fixed keyword set; adding
# a new rejection reason type upstream just needs one more line here.

_REASON_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "excessive_smoothing": ("smooth",),
    "missing_seasonality": ("season",),
    "missing_trend": ("trend",),
    "drift": ("drift", "wasserstein", "statistic", "threshold"),
    "elimination": ("eliminat",),
    "ranking": ("composite", "rank", "score"),
    "insufficient_history": ("insufficient", "history", "short"),
    "validation_failure": ("valid", "fail"),
}


def _reason_categories(text: str) -> set[str]:
    lowered = (text or "").lower()
    return {category for category, keywords in _REASON_CATEGORY_KEYWORDS.items() if any(k in lowered for k in keywords)}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "detail": self.detail}


# ---------------------------------------------------------------------
# Individual checks — each pure, each testable in isolation.
# ---------------------------------------------------------------------


def check_schema_validity(case: EvalCase, payload: InsightPayload | None, raw_text: str | None) -> CheckResult:
    """Reuses the existing v2 structured-output validator (`schema.
    parse_and_validate`) whenever the model's raw response is available —
    the same function `LLMInsightEngine` itself validates against, so this
    is not a second, competing schema check.

    When no raw text exists (the deterministic template path, which builds
    an already-typed `InsightPayload` directly), the payload is schema-valid
    by construction — there is nothing further to re-derive.
    """
    if raw_text is not None:
        try:
            parse_and_validate(raw_text, expected_model=case.selected_model)
        except SchemaValidationError as exc:
            return CheckResult("schema_validity", False, "; ".join(exc.problems))
        return CheckResult("schema_validity", True, "conforms to the v2 structured schema")

    if payload is None:
        return CheckResult("schema_validity", False, "no payload was produced")
    return CheckResult("schema_validity", True, "constructed via the schema-typed InsightPayload")


def check_winner_consistency(case: EvalCase, payload: InsightPayload) -> CheckResult:
    """PASS iff the LLM's `selected_model` matches the engine's real
    winner. The LLM must never describe a rejected model as the winner."""
    actual = (payload.selected_model or "").strip().lower()
    expected = case.selected_model.strip().lower()
    if actual == expected:
        return CheckResult("winner_consistency", True, "matches the engine's winner")
    return CheckResult(
        "winner_consistency", False,
        f"LLM said '{payload.selected_model}', engine winner was '{case.selected_model}'",
    )


def check_rejection_accuracy(case: EvalCase, payload: InsightPayload) -> CheckResult:
    """Compares the LLM's rejection reasons against the engine's real
    ones, by model + reason *category* rather than exact wording.

    PASS when the case has no rejected candidates at all — nothing to
    verify. Every rejected candidate the engine actually named must be
    both mentioned and match on at least one reason category somewhere in
    the LLM's response.

    Checked across `rejection_reasons` *and* `concise_summary` together,
    not `rejection_reasons` alone: the v2 schema's own field description
    is "one short phrase per rejected candidate" with no model-name field,
    and in practice the model reliably follows that literally — the
    category phrase goes in `rejection_reasons` ("missing seasonality")
    while the candidate's *name* goes in the prose sentence that
    introduces it ("...outperformed prophet, which was eliminated due to
    missing seasonality"). Requiring the name inside `rejection_reasons`
    itself would fail a well-formed, schema-compliant response for
    following the schema's own field shape — this was confirmed against a
    real Azure OpenAI run before being written this way, not assumed.
    """
    if not case.rejected_candidates:
        return CheckResult("rejection_accuracy", True, "no rejected candidates to verify")

    # A "local window" per candidate rather than one bag-of-words over the
    # whole response: every `rejection_reasons` entry (there is no reliable
    # way to know which candidate a given entry is *for*, since the schema
    # has no model-name field there) plus whichever `concise_summary`
    # sentence(s) actually name this candidate — not the full summary, so a
    # multi-candidate response can't pass one candidate's category check on
    # a category that only ever applied to a different candidate.
    summary_sentences = [s for s in _SENTENCE_SPLIT_RE.split(payload.concise_summary or "") if s.strip()]

    misses: list[str] = []
    for candidate in case.rejected_candidates:
        name = (candidate.get("model_name") or "").strip()
        reason = candidate.get("reason") or ""

        naming_sentences = [s for s in summary_sentences if name and name.lower() in s.lower()]
        window = " || ".join([*payload.rejection_reasons, *naming_sentences])

        if name and name.lower() not in window.lower():
            misses.append(f"{name}: not mentioned anywhere in the response")
            continue

        expected_categories = _reason_categories(reason)
        if expected_categories and not (expected_categories & _reason_categories(window)):
            misses.append(f"{name}: reason category mismatch (expected one of {sorted(expected_categories)})")

    if misses:
        return CheckResult("rejection_accuracy", False, "; ".join(misses))
    return CheckResult("rejection_accuracy", True, "every rejected candidate's reason matches its category")


def check_groundedness(case: EvalCase, payload: InsightPayload) -> tuple[CheckResult, str]:
    """Groundedness, plus an explicit hallucination category:

        grounded      — every claim traces to the supplied context
        contradictory — the LLM named the wrong model as winner (the most
                         serious failure: not just an unsupported number,
                         an outright wrong conclusion)
        unsupported    — a number/claim not present in the context

    Reuses `grounding.check_grounding` unmodified for the number-level
    check — this function only adds the winner-mismatch -> "contradictory"
    classification on top, which `check_grounding` (number-only) cannot see.
    """
    metrics: dict[str, float | int | None] = {
        "wmape": case.wmape,
        "confidence_estimate": case.confidence_pct,
        **case.extra_metrics,
    }
    grounding = check_grounding(payload.concise_summary, metrics)
    winner_ok = (payload.selected_model or "").strip().lower() == case.selected_model.strip().lower()

    if not winner_ok:
        category = "contradictory"
        passed = False
        detail = f"claims '{payload.selected_model}' won; the engine's real winner was '{case.selected_model}'"
    elif not grounding.grounded:
        category = "unsupported"
        passed = False
        detail = "; ".join(grounding.issues)
    else:
        category = "grounded"
        passed = True
        detail = "every claim traces to the supplied context"

    return CheckResult("groundedness", passed, detail), category


_JSON_LEAK_RE = re.compile(r'[{}]|"\s*:\s*"')
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def check_readability(case: EvalCase, payload: InsightPayload) -> CheckResult:
    """Simple, measurable rules — not a style judgement:

      * stays within the schema's own word cap (`MAX_SUMMARY_WORDS`)
      * the winner is named in the first sentence (v2 prompt: "Sentence 1
        MUST open by naming the model in selected_model")
      * the first sentence never opens with a rejected candidate's name
      * no raw JSON/technical formatting leaks into the text
      * no sentence is repeated verbatim
    """
    issues: list[str] = []
    text = (payload.concise_summary or "").strip()

    if not text:
        return CheckResult("readability", False, "summary is empty")

    words = text.split()
    if len(words) > MAX_SUMMARY_WORDS:
        issues.append(f"{len(words)} words exceeds the {MAX_SUMMARY_WORDS}-word cap")

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    first_sentence = sentences[0].lower() if sentences else text.lower()
    # "Opens with" the model, not merely "mentions it somewhere in sentence
    # 1" — the v2 prompt's rule is literally "Sentence 1 MUST open by
    # naming the model", so a sentence that only names it at the end
    # ("...before choosing xgboost") must still fail this.
    opening_words = " ".join(first_sentence.split()[:6])

    if case.selected_model.lower() not in opening_words:
        issues.append("selected model is not named at the start of the first sentence")

    for candidate in case.rejected_candidates:
        name = (candidate.get("model_name") or "").lower()
        if name and first_sentence.startswith(name):
            issues.append(f"summary opens with rejected candidate '{candidate.get('model_name')}'")

    if _JSON_LEAK_RE.search(text):
        issues.append("raw JSON/technical formatting leaked into the summary")

    lowered_sentences = [s.lower() for s in sentences]
    if len(lowered_sentences) != len(set(lowered_sentences)):
        issues.append("a sentence is repeated verbatim")

    return CheckResult("readability", not issues, "; ".join(issues) if issues else "reads cleanly")


# ---------------------------------------------------------------------
# Per-case orchestration
# ---------------------------------------------------------------------

_CHECK_ORDER = ("schema_validity", "groundedness", "winner_consistency", "rejection_accuracy", "readability")


@dataclass
class CaseEvalResult:
    case_id: str
    scenario: str | None
    payload: InsightPayload | None
    hallucination_category: str  # grounded | unsupported | contradictory | not_generated
    checks: list[CheckResult] = field(default_factory=list)
    # The case's own ground truth — carried through so a caller (the
    # Observability UI) can show "expected vs actual" without needing the
    # original `EvalCase` list kept in sync separately.
    expected: dict[str, Any] = field(default_factory=dict)
    # Set only when `generate()` itself raised — distinct from a schema-
    # invalid response, which is a real (graded) response, just a bad one.
    generation_error: str | None = None

    @property
    def overall_pass(self) -> bool:
        if self.generation_error or not self.checks:
            return False
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "scenario": self.scenario,
            "expected": self.expected,
            "insight": self.payload.to_dict() if self.payload else None,
            "hallucination_category": self.hallucination_category,
            "checks": {check.name: check.to_dict() for check in self.checks},
            "failed_checks": [check.name for check in self.checks if not check.passed],
            "generation_error": self.generation_error,
            "overall": "PASS" if self.overall_pass else "FAIL",
        }


def _expected_facts(case: EvalCase) -> dict[str, Any]:
    """The case's own ground truth, in the shape the UI shows as "Expected"
    — a subset of `EvalCase`'s fields, not a new structure."""
    return {
        "selected_model": case.selected_model,
        "wmape": case.wmape,
        "is_fallback": case.is_fallback,
        "fallback_trigger": case.fallback_trigger,
        "rejected_candidates": case.rejected_candidates,
        "confidence_pct": case.confidence_pct,
        "caveats": case.caveats,
    }


def evaluate_case(case: EvalCase, generate: GenerateFn) -> CaseEvalResult:
    """Run one case through `generate` and score every check.

    A raised exception is caught and recorded as `generation_error` — the
    same "a bad case must not stop the suite" contract `run_evaluation`
    already establishes in `evaluation.py`.
    """
    expected = _expected_facts(case)
    try:
        payload, raw_text = generate(case)
    except Exception as exc:  # noqa: BLE001 - a bad case must not stop the suite
        return CaseEvalResult(
            case_id=case.case_id, scenario=case.scenario, payload=None,
            hallucination_category="not_generated", checks=[], expected=expected,
            generation_error=f"{type(exc).__name__}: {exc}",
        )

    schema_check = check_schema_validity(case, payload, raw_text)

    if payload is None:
        checks = [
            schema_check,
            CheckResult("groundedness", False, "no payload was produced"),
            CheckResult("winner_consistency", False, "no payload was produced"),
            CheckResult("rejection_accuracy", False, "no payload was produced"),
            CheckResult("readability", False, "no payload was produced"),
        ]
        return CaseEvalResult(
            case_id=case.case_id, scenario=case.scenario, payload=None,
            hallucination_category="not_generated", checks=checks, expected=expected,
        )

    grounding_check, hallucination_category = check_groundedness(case, payload)
    checks = [
        schema_check,
        grounding_check,
        check_winner_consistency(case, payload),
        check_rejection_accuracy(case, payload),
        check_readability(case, payload),
    ]
    return CaseEvalResult(
        case_id=case.case_id, scenario=case.scenario, payload=payload, expected=expected,
        hallucination_category=hallucination_category, checks=checks,
    )


# ---------------------------------------------------------------------
# Thresholds + aggregate report
# ---------------------------------------------------------------------


@dataclass
class RegressionThresholds:
    """Configurable pass/fail gates over the whole evaluation set.

    Defaults are deliberately practical for a ~16-case set, not
    aspirational: with 16 cases, one miss on any single-case-sensitive
    metric is already ~6%, so a 100% bar on anything but winner
    consistency would fail on the first realistic wobble. Winner
    consistency is the one exception, set to 100%: the LLM naming the
    wrong model as winner is a correctness bug, not a quality nuance, and
    is never acceptable regardless of dataset size. Tune these down only
    with a documented reason as the dataset grows and stabilizes.
    """

    minimum_schema_pass_rate: float = 0.90
    minimum_groundedness: float = 0.85
    minimum_winner_consistency: float = 1.0
    minimum_rejection_accuracy: float = 0.80
    maximum_hallucination_rate: float = 0.15
    minimum_readability_pass_rate: float = 0.85

    @classmethod
    def default(cls) -> "RegressionThresholds":
        return cls()

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RegressionThresholds":
        if not payload:
            return cls()
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def to_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


@dataclass
class RegressionReport:
    dataset_version: str
    prompt_version: str
    generation_mode: str  # e.g. "azure_openai:gpt-4.1-mini" | "template"
    results: list[CaseEvalResult] = field(default_factory=list)
    thresholds: RegressionThresholds = field(default_factory=RegressionThresholds.default)

    @property
    def case_count(self) -> int:
        return len(self.results)

    def _graded(self) -> list[CaseEvalResult]:
        # Excludes only hard generation failures (an exception from
        # generate()) — a schema-invalid response is still a real, graded
        # response, and must still pull the rates down, not disappear.
        return [r for r in self.results if r.generation_error is None]

    @property
    def generated_count(self) -> int:
        return len(self._graded())

    def _check_rate(self, name: str) -> float | None:
        graded = self._graded()
        if not graded:
            return None
        passed = sum(1 for r in graded for c in r.checks if c.name == name and c.passed)
        return round(passed / len(graded), 4)

    @property
    def schema_pass_rate(self) -> float | None:
        return self._check_rate("schema_validity")

    @property
    def groundedness_rate(self) -> float | None:
        return self._check_rate("groundedness")

    @property
    def winner_consistency_rate(self) -> float | None:
        return self._check_rate("winner_consistency")

    @property
    def rejection_accuracy_rate(self) -> float | None:
        return self._check_rate("rejection_accuracy")

    @property
    def readability_pass_rate(self) -> float | None:
        return self._check_rate("readability")

    @property
    def hallucination_rate(self) -> float | None:
        graded = self._graded()
        if not graded:
            return None
        hallucinated = sum(1 for r in graded if r.hallucination_category not in ("grounded",))
        return round(hallucinated / len(graded), 4)

    @property
    def overall_pass_rate(self) -> float | None:
        graded = self._graded()
        if not graded:
            return None
        return round(sum(1 for r in graded if r.overall_pass) / len(graded), 4)

    def threshold_violations(self) -> list[str]:
        violations: list[str] = []
        gates: list[tuple[str, float | None, float, str]] = [
            ("schema_pass_rate", self.schema_pass_rate, self.thresholds.minimum_schema_pass_rate, "min"),
            ("groundedness_rate", self.groundedness_rate, self.thresholds.minimum_groundedness, "min"),
            ("winner_consistency_rate", self.winner_consistency_rate, self.thresholds.minimum_winner_consistency, "min"),
            ("rejection_accuracy_rate", self.rejection_accuracy_rate, self.thresholds.minimum_rejection_accuracy, "min"),
            ("readability_pass_rate", self.readability_pass_rate, self.thresholds.minimum_readability_pass_rate, "min"),
            ("hallucination_rate", self.hallucination_rate, self.thresholds.maximum_hallucination_rate, "max"),
        ]
        for name, actual, threshold, kind in gates:
            if actual is None:
                continue
            ok = actual >= threshold if kind == "min" else actual <= threshold
            if not ok:
                bound = "below minimum" if kind == "min" else "above maximum"
                violations.append(f"{name}={actual:.1%} is {bound} {threshold:.1%}")

        failed_generation = self.case_count - self.generated_count
        if failed_generation:
            violations.append(f"{failed_generation} case(s) failed to generate at all")

        return violations

    @property
    def regression_passed(self) -> bool:
        return not self.threshold_violations()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "prompt_version": self.prompt_version,
            "generation_mode": self.generation_mode,
            "case_count": self.case_count,
            "generated_count": self.generated_count,
            "schema_pass_rate": self.schema_pass_rate,
            "groundedness_rate": self.groundedness_rate,
            "winner_consistency_rate": self.winner_consistency_rate,
            "rejection_accuracy_rate": self.rejection_accuracy_rate,
            "hallucination_rate": self.hallucination_rate,
            "readability_pass_rate": self.readability_pass_rate,
            "overall_pass_rate": self.overall_pass_rate,
            "thresholds": self.thresholds.to_dict(),
            "regression_passed": self.regression_passed,
            "threshold_violations": self.threshold_violations(),
            "results": [r.to_dict() for r in self.results],
        }


def evaluate_regression_set(
    cases: list[EvalCase],
    generate: GenerateFn,
    *,
    dataset_version: str = "unknown",
    prompt_version: str = "v2",
    generation_mode: str = "unknown",
    thresholds: RegressionThresholds | None = None,
) -> RegressionReport:
    results = [evaluate_case(case, generate) for case in cases]
    return RegressionReport(
        dataset_version=dataset_version,
        prompt_version=prompt_version,
        generation_mode=generation_mode,
        results=results,
        thresholds=thresholds or RegressionThresholds.default(),
    )
