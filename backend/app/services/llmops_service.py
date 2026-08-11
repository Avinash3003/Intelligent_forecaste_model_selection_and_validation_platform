"""LLMOps observability — one run's LLM activity, reshaped for the UI.

Reads the exact same `PipelineExecutionResult` the Results dashboard and
the debug view read (`business_insights` for the structured per-group
insight, `llm_trace` for the detailed per-call record) and merges them into
one row per forecast group. No new storage, no new MLflow round trip: both
fields already travel through `summary.json`, which every Runner already
reads via `get_result()` — this module only reshapes what's already there.

A forecast group that never made a real LLM call (the template fallback
path, which `LLMInsightEngine` takes without touching Azure OpenAI at all)
still gets a row here, with `provider="template"` and no tokens/latency —
Section 13's LLMOps view is meant to show what happened to every key, not
only the ones that reached the network.
"""

from __future__ import annotations

from typing import Any

from app.orchestration.executor import PipelineExecutor, get_pipeline_executor
from app.schemas.llmops import LLMOpsAttempt, LLMOpsCall, LLMOpsResponse, LLMOpsSummary


class LLMOpsService:
    """Builds the LLMOps observability payload for one run."""

    def __init__(self, executor: PipelineExecutor | None = None) -> None:
        self._executor = executor or get_pipeline_executor()

    def get_llmops(self, run_id: str) -> LLMOpsResponse:
        """Assemble the LLMOps view.

        Never raises for a run with no LLM activity — a local run with
        Azure OpenAI disabled, or one still executing, simply has nothing
        to show yet, which `available=False` reports honestly rather than
        as an error. `UnknownRunError` still propagates: an unknown run_id
        is a 404, not an empty observability view.
        """
        result = self._executor.get_result(run_id)
        insights = result.business_insights or {}
        trace = result.llm_trace or {}

        groups_by_id: dict[str, dict[str, Any]] = insights.get("groups") or {}
        attempts_by_group = self._group_attempts(trace.get("calls") or [])

        # Every group either produced insight text or made at least one LLM
        # attempt (or both) — the union is every group this run has
        # anything to report on. A group present in neither has no
        # observability record, which is not the same as a failure.
        group_ids = list(dict.fromkeys([*groups_by_id.keys(), *attempts_by_group.keys()]))

        calls = [
            self._build_call(group_id, groups_by_id.get(group_id), attempts_by_group.get(group_id, []))
            for group_id in group_ids
        ]

        summary = self._build_summary(insights, trace.get("summary") or {})

        return LLMOpsResponse(
            run_id=run_id,
            available=bool(calls),
            summary=summary,
            calls=calls,
        )

    def _group_attempts(self, raw_calls: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for call in raw_calls:
            group_id = call.get("group_id")
            if not group_id:
                continue
            grouped.setdefault(group_id, []).append(call)
        return grouped

    def _build_call(
        self, group_id: str, insight: dict[str, Any] | None, raw_attempts: list[dict[str, Any]]
    ) -> LLMOpsCall:
        insight = insight or {}
        payload = insight.get("insight") or {}

        attempts = [self._build_attempt(raw) for raw in raw_attempts]
        # The last attempt is the one whose outcome actually stands — a
        # retried group's earlier attempts are history, not the result.
        final = attempts[-1] if attempts else None

        return LLMOpsCall(
            group_id=group_id,
            forecast_model=payload.get("selected_model"),
            provider=insight.get("provider") or (final.provider if final else "none"),
            deployment=final.deployment if final else None,
            prompt_version=insight.get("prompt_version"),
            timestamp=final.request_start_time if final else None,
            input_tokens=final.input_tokens if final else None,
            output_tokens=final.output_tokens if final else None,
            total_tokens=final.total_tokens if final else None,
            latency_ms=final.latency_ms if final else None,
            estimated_cost_usd=final.estimated_cost_usd if final else None,
            retry_count=insight.get("retry_count") or max(len(attempts) - 1, 0),
            validation_status=insight.get("validation_status") or (final.validation_status if final else "not_attempted"),
            grounding_status=insight.get("grounding_status") or (final.grounding_status if final else "not_attempted"),
            final_status=final.final_status if final else ("success" if payload else "pending"),
            error=insight.get("error") or (final.error if final else None),
            concise_summary=payload.get("concise_summary") or None,
            rejection_reasons=list(payload.get("rejection_reasons") or []),
            caveats=list(payload.get("caveats") or []),
            confidence=payload.get("confidence"),
            validation_errors=list(final.validation_errors) if final else [],
            grounding_issues=list(final.grounding_issues) if final else [],
            attempts=attempts,
        )

    def _build_attempt(self, raw: dict[str, Any]) -> LLMOpsAttempt:
        return LLMOpsAttempt(
            attempt_number=raw.get("attempt_number") or 1,
            provider=raw.get("provider") or "none",
            deployment=raw.get("deployment"),
            routing_tier=raw.get("routing_tier"),
            request_start_time=raw.get("request_start_time"),
            request_end_time=raw.get("request_end_time"),
            latency_ms=raw.get("latency_ms"),
            input_tokens=raw.get("prompt_tokens"),
            output_tokens=raw.get("completion_tokens"),
            total_tokens=raw.get("total_tokens"),
            estimated_cost_usd=raw.get("estimated_cost_usd"),
            validation_status=raw.get("validation_status") or "not_attempted",
            validation_errors=list(raw.get("validation_errors") or []),
            grounding_status=raw.get("grounding_status") or "not_attempted",
            grounding_issues=list(raw.get("grounding_issues") or []),
            final_status=raw.get("final_status") or "pending",
            error=raw.get("error"),
        )

    def _build_summary(self, insights: dict[str, Any], trace_summary: dict[str, Any]) -> LLMOpsSummary:
        return LLMOpsSummary(
            call_count=trace_summary.get("call_count") or 0,
            input_tokens=trace_summary.get("prompt_tokens") or 0,
            output_tokens=trace_summary.get("completion_tokens") or 0,
            total_tokens=trace_summary.get("total_tokens") or 0,
            average_latency_ms=trace_summary.get("average_latency_ms"),
            estimated_cost_usd=trace_summary.get("estimated_cost_usd"),
            cost_available=bool(trace_summary.get("cost_available")),
            groundedness_rate=trace_summary.get("groundedness_rate"),
            retry_count=trace_summary.get("retry_count") or 0,
            provider=insights.get("provider"),
            deployment=insights.get("model_name"),
            prompt_version=insights.get("prompt_version"),
            status=insights.get("status") or "not_generated",
        )


_service: LLMOpsService | None = None


def get_llmops_service() -> LLMOpsService:
    global _service
    if _service is None:
        _service = LLMOpsService()
    return _service
