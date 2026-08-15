"""Reads the LLM regression report the evaluate CLI writes.

Read-only: running an evaluation stays an explicit CLI action, never
triggered by an HTTP GET, and this never talks to Azure OpenAI directly.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.config.settings import Settings, get_settings
from app.schemas.llm_evaluation import LlmEvaluationResponse

logger = logging.getLogger(__name__)


class LlmEvaluationService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def get_latest_report(self) -> LlmEvaluationResponse:
        path = self._settings.llm_eval_report_path_resolved
        if not path.is_file():
            return LlmEvaluationResponse(
                available=False,
                unavailable_reason=(
                    "No evaluation report found yet. Run "
                    "'python -m forecast_engine.s11_llm.evaluate' to generate one."
                ),
            )

        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Could not read LLM evaluation report at '%s': %s", path, exc)
            return LlmEvaluationResponse(
                available=False,
                unavailable_reason="The evaluation report could not be read. Re-run the evaluation.",
            )

        generated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        return self._to_response(payload, generated_at)

    def _to_response(self, payload: dict[str, Any], generated_at: str) -> LlmEvaluationResponse:
        return LlmEvaluationResponse(
            available=True,
            dataset_version=payload.get("dataset_version"),
            prompt_version=payload.get("prompt_version"),
            generation_mode=payload.get("generation_mode"),
            case_count=payload.get("case_count") or 0,
            generated_count=payload.get("generated_count") or 0,
            schema_pass_rate=payload.get("schema_pass_rate"),
            groundedness_rate=payload.get("groundedness_rate"),
            winner_consistency_rate=payload.get("winner_consistency_rate"),
            rejection_accuracy_rate=payload.get("rejection_accuracy_rate"),
            hallucination_rate=payload.get("hallucination_rate"),
            readability_pass_rate=payload.get("readability_pass_rate"),
            overall_pass_rate=payload.get("overall_pass_rate"),
            thresholds=payload.get("thresholds"),
            regression_passed=payload.get("regression_passed"),
            threshold_violations=list(payload.get("threshold_violations") or []),
            results=payload.get("results") or [],
            generated_at=generated_at,
        )


_service: LlmEvaluationService | None = None


def get_llm_evaluation_service() -> LlmEvaluationService:
    global _service
    if _service is None:
        _service = LlmEvaluationService()
    return _service
