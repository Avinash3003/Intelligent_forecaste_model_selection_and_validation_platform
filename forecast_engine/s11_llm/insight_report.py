"""Business Insight result model — the output contract of Section 6.12.

Every field is narrative text produced by the LLM, kept strictly separate
from every deterministic report earlier in the pipeline: nothing here is
ever read back into a forecasting decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class BusinessInsightReport:
    """Structured LLM output — Section 6.12's OUTPUT list, one field each."""

    technical_explanation: str | None = None
    business_explanation: str | None = None
    forecast_summary: str | None = None
    winner_model_summary: str | None = None
    important_features: str | None = None
    forecast_risks: str | None = None
    drift_summary: str | None = None

    available: bool = False
    status: str = "not_generated"
    provider: str | None = None
    model_name: str | None = None
    section_errors: dict[str, str] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Serialize the insight report to a plain dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "technical_explanation": self.technical_explanation,
            "business_explanation": self.business_explanation,
            "forecast_summary": self.forecast_summary,
            "winner_model_summary": self.winner_model_summary,
            "important_features": self.important_features,
            "forecast_risks": self.forecast_risks,
            "drift_summary": self.drift_summary,
            "available": self.available,
            "status": self.status,
            "provider": self.provider,
            "model_name": self.model_name,
            "section_errors": self.section_errors,
            "generated_at": self.generated_at.isoformat(timespec="seconds"),
        }
