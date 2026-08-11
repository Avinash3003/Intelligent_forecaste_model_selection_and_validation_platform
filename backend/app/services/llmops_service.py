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

from app.orchestration.exceptions import UnknownRunError
from app.orchestration.executor import PipelineExecutor, get_pipeline_executor
from app.orchestration.schemas import JobStatus
from app.schemas.llmops import (
    LLMOpsAttempt,
    LLMOpsCall,
    LLMOpsResponse,
    LLMOpsSummary,
    PromptUsageResponse,
    PromptVersionUsage,
)


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

    def get_prompt_usage(self) -> PromptUsageResponse:
        """LLM usage/performance/quality across every completed run,
        grouped by the prompt version each run actually used.

        Built entirely from repeated `get_llmops()` calls — the exact same
        per-run read this module already does, run once per completed run
        instead of once. No second trace format, no new storage: a prompt
        version's numbers are just its runs' numbers, summed.
        """
        accumulators: dict[str, _PromptVersionAccumulator] = {}

        for listing in self._executor.list_runs():
            if listing.job_status is not JobStatus.COMPLETED:
                continue
            try:
                response = self.get_llmops(listing.run_id)
            except UnknownRunError:
                # Listed a moment ago, gone now (e.g. store reset) — the
                # aggregate simply does not include it, same as it would
                # not include a run nobody had ever submitted.
                continue
            if not response.available:
                continue

            version = response.summary.prompt_version or "unknown"
            accumulators.setdefault(version, _PromptVersionAccumulator(version)).add_run(response)

        versions = sorted(
            (acc.finalize() for acc in accumulators.values()),
            key=lambda v: v.call_count,
            reverse=True,
        )
        return PromptUsageResponse(versions=versions)

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


class _PromptVersionAccumulator:
    """Running totals for one prompt version, across however many
    completed runs used it.

    Usage/performance figures (calls, tokens, cost, latency) come from
    each run's own `summary` — already the exact count of real network
    attempts, correctly computed once per run. Quality/reliability figures
    (groundedness, validation, retry, fallback) come from each run's
    `calls` instead — one entry per *forecast group*, carrying that
    group's *final* outcome after any retries — because those are
    properties of "did this key end up with a good, grounded answer",
    not of the raw request count, and a template-only group (no network
    attempt at all) still needs to count toward "how often did the system
    fall back" even though it contributes zero to the call/token totals.
    """

    def __init__(self, prompt_version: str) -> None:
        self.prompt_version = prompt_version
        self.runs_included = 0

        self.call_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self._latency_ms_sum = 0.0
        self._latency_weighted_calls = 0
        self._cost_sum = 0.0
        self._cost_known_for_all_runs = True
        self._any_run_had_calls = False

        self._grounded = 0
        self._ungrounded = 0
        self._validation_passed = 0
        self._validation_failed = 0
        self._groups_total = 0
        self._groups_retried = 0
        self._groups_fallback = 0

    def add_run(self, response: LLMOpsResponse) -> None:
        self.runs_included += 1
        summary = response.summary

        self.call_count += summary.call_count
        self.input_tokens += summary.input_tokens
        self.output_tokens += summary.output_tokens
        self.total_tokens += summary.total_tokens

        if summary.call_count:
            self._any_run_had_calls = True
            if summary.average_latency_ms is not None:
                # Recovers this run's real latency sum from its average,
                # so combining runs of different sizes weights correctly
                # instead of averaging averages.
                self._latency_ms_sum += summary.average_latency_ms * summary.call_count
                self._latency_weighted_calls += summary.call_count

            if summary.cost_available and summary.estimated_cost_usd is not None:
                self._cost_sum += summary.estimated_cost_usd
            else:
                self._cost_known_for_all_runs = False

        for call in response.calls:
            self._groups_total += 1
            if call.grounding_status == "grounded":
                self._grounded += 1
            elif call.grounding_status == "ungrounded":
                self._ungrounded += 1
            if call.validation_status == "passed":
                self._validation_passed += 1
            elif call.validation_status == "failed":
                self._validation_failed += 1
            if call.retry_count > 0:
                self._groups_retried += 1
            if call.provider not in ("azure_openai",):
                self._groups_fallback += 1

    def finalize(self) -> PromptVersionUsage:
        graded = self._grounded + self._ungrounded
        validated = self._validation_passed + self._validation_failed
        cost_available = self._any_run_had_calls and self._cost_known_for_all_runs

        return PromptVersionUsage(
            prompt_version=self.prompt_version,
            runs_included=self.runs_included,
            call_count=self.call_count,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            estimated_cost_usd=round(self._cost_sum, 6) if cost_available else None,
            cost_available=cost_available,
            average_latency_ms=(
                round(self._latency_ms_sum / self._latency_weighted_calls, 1)
                if self._latency_weighted_calls
                else None
            ),
            groundedness_rate=round(self._grounded / graded, 4) if graded else None,
            validation_pass_rate=round(self._validation_passed / validated, 4) if validated else None,
            retry_rate=round(self._groups_retried / self._groups_total, 4) if self._groups_total else None,
            fallback_rate=round(self._groups_fallback / self._groups_total, 4) if self._groups_total else None,
        )


_service: LLMOpsService | None = None


def get_llmops_service() -> LLMOpsService:
    global _service
    if _service is None:
        _service = LLMOpsService()
    return _service
