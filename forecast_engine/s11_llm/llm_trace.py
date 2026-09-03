"""One record per LLM call, detailed enough to debug any explanation.

MLflow's general-purpose artifact logging is not built for this granularity,
so this is a dedicated store.

In-process and JSON-serializable rather than a new persistence system: the
existing artifact pipeline already logs arbitrary JSON, so a store that
produces a plain dict slots straight into it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class LLMCallTrace:
    """Everything about one LLM call, end to end.

    One of these is recorded for every attempt — including a failed or
    retried one — so `retry_count`/`final_status` on the group's *last*
    trace tells the whole story, and every earlier attempt is still there
    for debugging.
    """

    run_id: str
    group_id: str
    model_name: str
    prompt_version: str
    deployment: str | None
    routing_tier: str

    request_start_time: datetime
    request_end_time: datetime | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    estimated_cost_usd: float | None = None

    validation_status: str = "not_attempted"  # not_attempted | passed | failed
    validation_errors: list[str] = field(default_factory=list)
    grounding_status: str = "not_attempted"  # not_attempted | grounded | ungrounded | skipped
    grounding_issues: list[str] = field(default_factory=list)

    attempt_number: int = 1
    final_status: str = "pending"  # pending | success | validation_failed | provider_error | fallback_used

    provider: str = "azure_openai"  # azure_openai | azure_openai_fallback | template
    error: str | None = None

    @property
    def latency_ms(self) -> float | None:
        if self.request_end_time is None:
            return None
        return (self.request_end_time - self.request_start_time).total_seconds() * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "group_id": self.group_id,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "deployment": self.deployment,
            "routing_tier": self.routing_tier,
            "request_start_time": self.request_start_time.isoformat(timespec="milliseconds"),
            "request_end_time": (
                self.request_end_time.isoformat(timespec="milliseconds") if self.request_end_time else None
            ),
            "latency_ms": round(self.latency_ms, 1) if self.latency_ms is not None else None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": (
                round(self.estimated_cost_usd, 6) if self.estimated_cost_usd is not None else None
            ),
            "validation_status": self.validation_status,
            "validation_errors": self.validation_errors,
            "grounding_status": self.grounding_status,
            "grounding_issues": self.grounding_issues,
            "attempt_number": self.attempt_number,
            "final_status": self.final_status,
            "provider": self.provider,
            "error": self.error,
        }


class LLMTraceStore:
    """Collects every `LLMCallTrace` made during one run."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._traces: list[LLMCallTrace] = []
        # Groups are generated concurrently, so every mutation of the list
        # and every read that walks it happens under this.
        self._lock = threading.Lock()

    def start_call(
        self, *, group_id: str, model_name: str, prompt_version: str, deployment: str | None, routing_tier: str,
        attempt_number: int = 1,
    ) -> LLMCallTrace:
        trace = LLMCallTrace(
            run_id=self._run_id,
            group_id=group_id,
            model_name=model_name,
            prompt_version=prompt_version,
            deployment=deployment,
            routing_tier=routing_tier,
            request_start_time=datetime.now(timezone.utc),
            attempt_number=attempt_number,
        )
        with self._lock:
            self._traces.append(trace)
        return trace

    def finish_call(self, trace: LLMCallTrace) -> None:
        trace.request_end_time = datetime.now(timezone.utc)

    @property
    def traces(self) -> list[LLMCallTrace]:
        with self._lock:
            return list(self._traces)

    def summary(self) -> dict[str, Any]:
        """Aggregate figures for the run-level LLM panel.

        This is exactly the "LLM Calls / tokens / latency / cost /
        groundedness / retries" block the phase's expected output names —
        computed once here so the backend and the run summary read it
        identically rather than each re-deriving it.
        """
        traces = self.traces
        if not traces:
            return {
                "call_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "average_latency_ms": None,
                "estimated_cost_usd": None,
                "cost_available": False,
                "retry_count": 0,
                "grounded_count": 0,
                "ungrounded_count": 0,
                "groundedness_rate": None,
                "prompt_versions": [],
                "final_statuses": {},
            }

        latencies = [t.latency_ms for t in traces if t.latency_ms is not None]
        costs = [t.estimated_cost_usd for t in traces if t.estimated_cost_usd is not None]
        cost_available = len(costs) == len([t for t in traces if t.total_tokens])

        grounded = sum(1 for t in traces if t.grounding_status == "grounded")
        ungrounded = sum(1 for t in traces if t.grounding_status == "ungrounded")
        graded = grounded + ungrounded

        final_statuses: dict[str, int] = {}
        for trace in traces:
            final_statuses[trace.final_status] = final_statuses.get(trace.final_status, 0) + 1

        return {
            "call_count": len(traces),
            "prompt_tokens": sum(t.prompt_tokens or 0 for t in traces),
            "completion_tokens": sum(t.completion_tokens or 0 for t in traces),
            "total_tokens": sum(t.total_tokens or 0 for t in traces),
            "average_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "estimated_cost_usd": round(sum(costs), 6) if cost_available and costs else None,
            "cost_available": cost_available and bool(costs),
            # A "retry" is any attempt beyond the first for the same group.
            "retry_count": sum(1 for t in traces if t.attempt_number > 1),
            "grounded_count": grounded,
            "ungrounded_count": ungrounded,
            "groundedness_rate": round(grounded / graded, 4) if graded else None,
            "prompt_versions": sorted({t.prompt_version for t in traces}),
            "final_statuses": final_statuses,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self._run_id, "summary": self.summary(), "calls": [t.to_dict() for t in self.traces]}


def elapsed_ms(started_at: float) -> float:
    """Milliseconds since `started_at` (a `time.perf_counter()` reading)."""
    return (time.perf_counter() - started_at) * 1000.0
