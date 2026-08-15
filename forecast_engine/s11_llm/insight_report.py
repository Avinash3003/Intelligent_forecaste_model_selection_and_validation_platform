"""What the insight stage produces.

GroupInsight wraps one group's validated payload plus how it was produced
(provider, prompt version, retries, grounding and validation status).
BusinessInsightReport is the run-level envelope: one per group, plus the
run's aggregate trace summary, so a caller never walks every call just to
total the tokens.

Nothing here is ever read back into a forecasting decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from forecast_engine.s11_llm.schema import InsightPayload


@dataclass
class GroupInsight:
    """One forecast group's structured explanation, plus its provenance."""

    group_id: str
    payload: InsightPayload | None
    provider: str  # azure_openai | azure_openai_fallback | template | none
    prompt_version: str
    validation_status: str  # passed | failed
    grounding_status: str  # grounded | ungrounded | skipped
    retry_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "insight": self.payload.to_dict() if self.payload else None,
            "provider": self.provider,
            "prompt_version": self.prompt_version,
            "validation_status": self.validation_status,
            "grounding_status": self.grounding_status,
            "retry_count": self.retry_count,
            "error": self.error,
        }


@dataclass
class BusinessInsightReport:
    """Structured LLM output — one `GroupInsight` per forecast group."""

    groups: dict[str, GroupInsight] = field(default_factory=dict)

    available: bool = False
    status: str = "not_generated"
    provider: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    section_errors: dict[str, str] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # The run's aggregate LLM trace summary (from `LLMTraceStore.summary()`)
    # — calls, tokens, latency, cost, groundedness, retries, budget usage.
    # Carried here (rather than only on a separate trace store the caller
    # has to remember to also serialize) so it travels with the run summary
    # for free, and reaches MLflow/the backend through the same path every
    # other report field already does.
    trace_summary: dict[str, Any] = field(default_factory=dict)

    # Section 13.2's per-run token budget: what was configured, what was
    # actually spent, and whether the budget was exhausted before every
    # group got an insight (in which case the remaining groups fell back
    # to the template path — see `insight_engine.py`).
    token_budget: int | None = None
    token_budget_exhausted: bool = False

    # Serialize the insight report to a plain dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": {group_id: insight.to_dict() for group_id, insight in self.groups.items()},
            "available": self.available,
            "status": self.status,
            "provider": self.provider,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "section_errors": self.section_errors,
            "generated_at": self.generated_at.isoformat(timespec="seconds"),
            "trace_summary": self.trace_summary,
            "token_budget": self.token_budget,
            "token_budget_exhausted": self.token_budget_exhausted,
        }

    def get(self, group_id: str) -> GroupInsight | None:
        return self.groups.get(group_id)
