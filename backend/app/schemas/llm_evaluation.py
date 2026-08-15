"""The HTTP shape of the JSON the LLM evaluate CLI writes.

Mirrors the engine's own report field-for-field rather than reshaping it:
the engine is the source of truth, this is only its contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LlmEvalCheckResult(BaseModel):
    passed: bool
    detail: str = ""


class LlmEvalCaseResult(BaseModel):
    case_id: str
    scenario: str | None = None
    expected: dict = Field(default_factory=dict)
    insight: dict | None = None
    hallucination_category: str  # grounded | unsupported | contradictory | not_generated
    checks: dict[str, LlmEvalCheckResult] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)
    generation_error: str | None = None
    overall: str  # PASS | FAIL


class LlmEvalThresholds(BaseModel):
    minimum_schema_pass_rate: float
    minimum_groundedness: float
    minimum_winner_consistency: float
    minimum_rejection_accuracy: float
    maximum_hallucination_rate: float
    minimum_readability_pass_rate: float


class LlmEvaluationResponse(BaseModel):
    """The full evaluation/regression report, or an honest "not yet run"
    when no report file exists — never a fabricated/placeholder result.
    """

    available: bool = False
    # Present only when `available` is False — why there is nothing to show.
    unavailable_reason: str | None = None

    dataset_version: str | None = None
    prompt_version: str | None = None
    generation_mode: str | None = None
    case_count: int = 0
    generated_count: int = 0

    schema_pass_rate: float | None = None
    groundedness_rate: float | None = None
    winner_consistency_rate: float | None = None
    rejection_accuracy_rate: float | None = None
    hallucination_rate: float | None = None
    readability_pass_rate: float | None = None
    overall_pass_rate: float | None = None

    thresholds: LlmEvalThresholds | None = None
    regression_passed: bool | None = None
    threshold_violations: list[str] = Field(default_factory=list)

    results: list[LlmEvalCaseResult] = Field(default_factory=list)

    # When the report file was generated (its own filesystem mtime) — lets
    # the UI show "as of ..." rather than implying this is live/real-time.
    generated_at: str | None = None
