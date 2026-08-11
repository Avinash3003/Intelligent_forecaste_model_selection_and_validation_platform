"""Prompt Usage & Performance — LLM usage/performance/quality aggregated
across completed runs, grouped by prompt version.

Built entirely from repeated `LLMOpsService.get_llmops()` calls, the exact
per-run read the existing per-run LLMOps view already does — these tests
cover the aggregation, not a new trace format.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_principal
from app.auth.models import Principal, Role
from app.auth.rbac import permissions_for
from app.main import app
from app.orchestration.exceptions import UnknownRunError
from app.orchestration.schemas import (
    ExecutionBackend,
    JobStatus,
    PipelineExecutionResult,
    RunListing,
)
from app.services.llmops_service import LLMOpsService, get_llmops_service


class _FakeExecutor:
    """Stands in for `PipelineExecutor`: a fixed run listing, each backed
    by a canned result (or missing, to prove a since-vanished run is
    skipped rather than raised)."""

    def __init__(self, listings: list[RunListing], results: dict[str, PipelineExecutionResult]) -> None:
        self._listings = listings
        self._results = results

    def list_runs(self) -> list[RunListing]:
        return self._listings

    def get_result(self, run_id: str) -> PipelineExecutionResult:
        if run_id not in self._results:
            raise UnknownRunError(f"No run found for run_id '{run_id}'.")
        return self._results[run_id]


def _listing(run_id: str, status: JobStatus = JobStatus.COMPLETED) -> RunListing:
    return RunListing(
        run_id=run_id,
        job_status=status,
        execution_backend=ExecutionBackend.LOCAL,
        started_at="2026-01-01T00:00:00",
    )


def _call(group_id, provider="azure_openai", grounding="grounded", validation="passed", retry_count=0):
    return {
        "group_id": group_id,
        "insight": {
            "selected_model": "prophet",
            "rejection_reasons": [],
            "confidence": 80.0,
            "caveats": [],
            "concise_summary": "x",
        },
        "provider": provider,
        "prompt_version": "v2",
        "validation_status": validation,
        "grounding_status": grounding,
        "retry_count": retry_count,
        "error": None,
    }


def _trace_call(group_id, provider="azure_openai", prompt_tokens=100, completion_tokens=20, latency_ms=1000.0, cost=None):
    return {
        "run_id": "run",
        "group_id": group_id,
        "model_name": "prophet",
        "prompt_version": "v2",
        "deployment": "gpt-4.1-mini",
        "routing_tier": "simple",
        "request_start_time": "2026-01-01T00:00:00",
        "request_end_time": "2026-01-01T00:00:01",
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": cost,
        "validation_status": "passed",
        "validation_errors": [],
        "grounding_status": "grounded",
        "grounding_issues": [],
        "attempt_number": 1,
        "final_status": "success",
        "provider": provider,
        "error": None,
    }


def _result(
    run_id: str,
    prompt_version: str = "v2",
    groups: list[dict] | None = None,
    trace_calls: list[dict] | None = None,
    cost_available: bool = False,
    cost_usd: float | None = None,
) -> PipelineExecutionResult:
    groups = groups or []
    trace_calls = trace_calls if trace_calls is not None else []
    call_count = len(trace_calls)
    total_tokens = sum(c["total_tokens"] for c in trace_calls)
    prompt_tokens = sum(c["prompt_tokens"] for c in trace_calls)
    completion_tokens = sum(c["completion_tokens"] for c in trace_calls)
    latencies = [c["latency_ms"] for c in trace_calls if c.get("latency_ms") is not None]

    business_insights = {
        "provider": "azure_openai",
        "model_name": "gpt-4.1-mini",
        "prompt_version": prompt_version,
        "status": "generated",
        "groups": {g["group_id"]: g for g in groups},
        "trace_summary": {
            "call_count": call_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "average_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "estimated_cost_usd": cost_usd,
            "cost_available": cost_available,
        },
    }
    llm_trace = {"run_id": run_id, "summary": business_insights["trace_summary"], "calls": trace_calls}

    return PipelineExecutionResult(
        run_id=run_id,
        job_status=JobStatus.COMPLETED,
        execution_backend=ExecutionBackend.LOCAL,
        business_insights=business_insights,
        llm_trace=llm_trace,
    )


def _service(listings, results) -> LLMOpsService:
    return LLMOpsService(_FakeExecutor(listings, results))


# ---------------------------------------------------------------------
# Grouping and basic aggregation
# ---------------------------------------------------------------------


def test_calls_and_tokens_are_summed_across_runs_of_the_same_version():
    results = {
        "run-1": _result(
            "run-1",
            groups=[_call("1 | 1")],
            trace_calls=[_trace_call("1 | 1", prompt_tokens=100, completion_tokens=20)],
        ),
        "run-2": _result(
            "run-2",
            groups=[_call("2 | 1")],
            trace_calls=[_trace_call("2 | 1", prompt_tokens=50, completion_tokens=10)],
        ),
    }
    usage = _service([_listing("run-1"), _listing("run-2")], results).get_prompt_usage()

    assert len(usage.versions) == 1
    v2 = usage.versions[0]
    assert v2.prompt_version == "v2"
    assert v2.runs_included == 2
    assert v2.call_count == 2
    assert v2.input_tokens == 150
    assert v2.output_tokens == 30
    assert v2.total_tokens == 180


def test_different_prompt_versions_are_kept_separate():
    results = {
        "run-1": _result("run-1", prompt_version="v1", groups=[_call("1 | 1")], trace_calls=[_trace_call("1 | 1")]),
        "run-2": _result("run-2", prompt_version="v2", groups=[_call("1 | 1")], trace_calls=[_trace_call("1 | 1")]),
    }
    usage = _service([_listing("run-1"), _listing("run-2")], results).get_prompt_usage()

    versions = {v.prompt_version for v in usage.versions}
    assert versions == {"v1", "v2"}
    assert all(v.runs_included == 1 for v in usage.versions)


# ---------------------------------------------------------------------
# Quality metrics — computed from final group outcomes
# ---------------------------------------------------------------------


def test_groundedness_and_validation_rates_are_computed_correctly():
    results = {
        "run-1": _result(
            "run-1",
            groups=[
                _call("1 | 1", grounding="grounded", validation="passed"),
                _call("1 | 2", grounding="ungrounded", validation="failed"),
                _call("1 | 3", grounding="grounded", validation="passed"),
            ],
        ),
    }
    usage = _service([_listing("run-1")], results).get_prompt_usage()

    v2 = usage.versions[0]
    assert v2.groundedness_rate == pytest.approx(2 / 3, abs=1e-4)
    assert v2.validation_pass_rate == pytest.approx(2 / 3, abs=1e-4)


def test_retry_and_fallback_rates_are_computed_from_groups_not_raw_calls():
    """A template-only group makes zero real network calls but must still
    count toward the fallback rate — it is exactly what 'fell back' means."""
    results = {
        "run-1": _result(
            "run-1",
            groups=[
                _call("1 | 1", provider="azure_openai", retry_count=0),
                _call("1 | 2", provider="template", retry_count=0),
                _call("1 | 3", provider="azure_openai", retry_count=2),
            ],
        ),
    }
    usage = _service([_listing("run-1")], results).get_prompt_usage()

    v2 = usage.versions[0]
    assert v2.fallback_rate == pytest.approx(1 / 3, abs=1e-4)
    assert v2.retry_rate == pytest.approx(1 / 3, abs=1e-4)


# ---------------------------------------------------------------------
# Cost — never fabricated
# ---------------------------------------------------------------------


def test_cost_is_summed_when_every_contributing_run_has_pricing():
    results = {
        "run-1": _result("run-1", groups=[_call("1 | 1")], trace_calls=[_trace_call("1 | 1")], cost_available=True, cost_usd=0.10),
        "run-2": _result("run-2", groups=[_call("2 | 1")], trace_calls=[_trace_call("2 | 1")], cost_available=True, cost_usd=0.05),
    }
    usage = _service([_listing("run-1"), _listing("run-2")], results).get_prompt_usage()

    v2 = usage.versions[0]
    assert v2.cost_available is True
    assert v2.estimated_cost_usd == pytest.approx(0.15)


def test_cost_is_na_when_any_contributing_run_lacks_pricing():
    """Never a partial sum presented as if it were the whole cost."""
    results = {
        "run-1": _result("run-1", groups=[_call("1 | 1")], trace_calls=[_trace_call("1 | 1")], cost_available=True, cost_usd=0.10),
        "run-2": _result("run-2", groups=[_call("2 | 1")], trace_calls=[_trace_call("2 | 1")], cost_available=False),
    }
    usage = _service([_listing("run-1"), _listing("run-2")], results).get_prompt_usage()

    v2 = usage.versions[0]
    assert v2.cost_available is False
    assert v2.estimated_cost_usd is None


def test_latency_is_call_count_weighted_not_a_naive_average_of_averages():
    results = {
        "run-1": _result(
            "run-1", groups=[_call(f"1 | {i}") for i in range(9)],
            trace_calls=[_trace_call(f"1 | {i}", latency_ms=1000.0) for i in range(9)],
        ),
        "run-2": _result(
            "run-2", groups=[_call("2 | 1")],
            trace_calls=[_trace_call("2 | 1", latency_ms=10000.0)],
        ),
    }
    usage = _service([_listing("run-1"), _listing("run-2")], results).get_prompt_usage()

    v2 = usage.versions[0]
    # 9 calls at 1000ms + 1 call at 10000ms = 19000ms / 10 calls = 1900ms,
    # not (1000 + 10000) / 2 = 5500ms.
    assert v2.average_latency_ms == pytest.approx(1900.0, abs=0.1)


# ---------------------------------------------------------------------
# Skips: non-completed runs, runs with no LLM activity, vanished runs
# ---------------------------------------------------------------------


def test_non_completed_runs_are_excluded():
    results = {"run-1": _result("run-1", groups=[_call("1 | 1")], trace_calls=[_trace_call("1 | 1")])}
    usage = _service([_listing("run-1", status=JobStatus.RUNNING)], results).get_prompt_usage()

    assert usage.versions == []


def test_runs_with_no_llm_activity_are_excluded_not_counted_as_zero():
    results = {"run-1": _result("run-1", groups=[], trace_calls=[])}
    usage = _service([_listing("run-1")], results).get_prompt_usage()

    assert usage.versions == []


def test_a_run_that_vanishes_between_listing_and_read_is_skipped_not_an_error():
    usage = _service([_listing("run-1")], {}).get_prompt_usage()

    assert usage.versions == []


def test_no_runs_at_all_returns_an_empty_list():
    usage = _service([], {}).get_prompt_usage()

    assert usage.versions == []


# ---------------------------------------------------------------------
# Endpoint: auth, and it's really additive (existing per-run route untouched)
# ---------------------------------------------------------------------


def _principal(role: Role) -> Principal:
    return Principal(subject=f"user-{role.value}", display_name=role.value, roles=[role], permissions=permissions_for([role]))


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_endpoint_requires_model_inspect_permission(client):
    app.dependency_overrides[get_current_principal] = lambda: _principal(Role.ANALYST)
    app.dependency_overrides[get_llmops_service] = lambda: _service([], {})

    assert client.get("/results/llmops/prompt-usage").status_code == 403


def test_endpoint_returns_aggregated_versions(client):
    app.dependency_overrides[get_current_principal] = lambda: _principal(Role.DATA_SCIENTIST)
    results = {"run-1": _result("run-1", groups=[_call("1 | 1")], trace_calls=[_trace_call("1 | 1")])}
    app.dependency_overrides[get_llmops_service] = lambda: _service([_listing("run-1")], results)

    response = client.get("/results/llmops/prompt-usage")

    assert response.status_code == 200
    body = response.json()
    assert body["versions"][0]["prompt_version"] == "v2"


def test_the_prompt_usage_path_does_not_shadow_the_per_run_llmops_route(client):
    """The literal '/llmops/prompt-usage' path must never be captured by
    the '/{run_id}/llmops' pattern or vice versa."""
    app.dependency_overrides[get_current_principal] = lambda: _principal(Role.ADMIN)
    results = {"run-1": _result("run-1", groups=[_call("1 | 1")], trace_calls=[_trace_call("1 | 1")])}
    app.dependency_overrides[get_llmops_service] = lambda: _service([_listing("run-1")], results)

    per_run = client.get("/results/run-1/llmops")
    aggregate = client.get("/results/llmops/prompt-usage")

    assert per_run.status_code == 200
    assert "versions" not in per_run.json()
    assert aggregate.status_code == 200
    assert "calls" not in aggregate.json()
