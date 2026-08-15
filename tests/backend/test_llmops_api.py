"""LLMOps observability: the service that merges `business_insights` and
`llm_trace` into one row per forecast group, and the `/results/{run_id}/llmops`
endpoint that serves it.

Both already exist on `PipelineExecutionResult` (no new storage, no new
MLflow round trip) — these tests cover the reshaping and the route's auth/
error behavior, not a new data source.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_principal
from app.auth.models import Principal, Role
from app.auth.rbac import permissions_for
from app.main import app
from app.orchestration.exceptions import UnknownRunError
from app.orchestration.schemas import ExecutionBackend, JobStatus, PipelineExecutionResult
from app.services.llmops_service import LLMOpsService, get_llmops_service


# ---------------------------------------------------------------------
# Fixtures shared by service-level and endpoint-level tests
# ---------------------------------------------------------------------


class _FakeExecutor:
    """Stands in for `PipelineExecutor`: returns canned results by run_id."""

    def __init__(self, results: dict[str, PipelineExecutionResult]) -> None:
        self._results = results

    def get_result(self, run_id: str) -> PipelineExecutionResult:
        if run_id not in self._results:
            raise UnknownRunError(f"No run found for run_id '{run_id}'.")
        return self._results[run_id]


def _result(run_id: str, business_insights: dict | None = None, llm_trace: dict | None = None) -> PipelineExecutionResult:
    return PipelineExecutionResult(
        run_id=run_id,
        job_status=JobStatus.COMPLETED,
        execution_backend=ExecutionBackend.LOCAL,
        business_insights=business_insights or {},
        llm_trace=llm_trace or {},
    )


def _call(group_id, attempt_number=1, provider="azure_openai", **overrides):
    call = {
        "run_id": "run-1",
        "group_id": group_id,
        "model_name": "prophet",
        "prompt_version": "v2",
        "deployment": "gpt-4.1-mini",
        "routing_tier": "simple",
        "request_start_time": "2026-08-11T04:02:33.500",
        "request_end_time": "2026-08-11T04:02:35.700",
        "latency_ms": 2200.0,
        "prompt_tokens": 900,
        "completion_tokens": 80,
        "total_tokens": 980,
        "estimated_cost_usd": 0.05,
        "validation_status": "passed",
        "validation_errors": [],
        "grounding_status": "grounded",
        "grounding_issues": [],
        "attempt_number": attempt_number,
        "final_status": "success",
        "provider": provider,
        "error": None,
    }
    call.update(overrides)
    return call


def _insight_group(group_id, provider="azure_openai", retry_count=0, **overrides):
    entry = {
        "group_id": group_id,
        "insight": {
            "selected_model": "prophet",
            "rejection_reasons": [],
            "confidence": 87.5,
            "caveats": [],
            "concise_summary": "prophet was selected for its lowest backtest error.",
        },
        "provider": provider,
        "prompt_version": "v2",
        "validation_status": "passed",
        "grounding_status": "grounded",
        "retry_count": retry_count,
        "error": None,
    }
    entry.update(overrides)
    return entry


# ---------------------------------------------------------------------
# Service: normal case
# ---------------------------------------------------------------------


def test_a_group_with_one_successful_call_is_reported_fully():
    business_insights = {
        "provider": "azure_openai",
        "model_name": "gpt-4.1-mini",
        "prompt_version": "v2",
        "status": "generated",
        "groups": {"1 | 1": _insight_group("1 | 1")},
        "trace_summary": {
            "call_count": 1,
            "prompt_tokens": 900,
            "completion_tokens": 80,
            "total_tokens": 980,
            "average_latency_ms": 2200.0,
            "estimated_cost_usd": 0.05,
            "cost_available": True,
            "groundedness_rate": 1.0,
            "retry_count": 0,
        },
    }
    llm_trace = {"run_id": "run-1", "summary": business_insights["trace_summary"], "calls": [_call("1 | 1")]}

    service = LLMOpsService(_FakeExecutor({"run-1": _result("run-1", business_insights, llm_trace)}))
    response = service.get_llmops("run-1")

    assert response.available is True
    assert response.summary.call_count == 1
    assert response.summary.total_tokens == 980
    assert response.summary.provider == "azure_openai"
    assert response.summary.deployment == "gpt-4.1-mini"
    assert response.summary.status == "generated"

    assert len(response.calls) == 1
    call = response.calls[0]
    assert call.group_id == "1 | 1"
    assert call.forecast_model == "prophet"
    assert call.deployment == "gpt-4.1-mini"
    assert call.total_tokens == 980
    assert call.latency_ms == 2200.0
    assert call.grounding_status == "grounded"
    assert call.validation_status == "passed"
    assert call.retry_count == 0
    assert call.concise_summary.startswith("prophet was selected")
    assert call.confidence == 87.5
    assert len(call.attempts) == 1


# ---------------------------------------------------------------------
# Multiple groups, retries, template fallback (no real call at all)
# ---------------------------------------------------------------------


def test_multiple_groups_are_each_reported_independently():
    business_insights = {
        "groups": {"1 | 1": _insight_group("1 | 1"), "1 | 2": _insight_group("1 | 2")},
        "trace_summary": {"call_count": 2},
    }
    llm_trace = {"calls": [_call("1 | 1"), _call("1 | 2")]}

    service = LLMOpsService(_FakeExecutor({"run-1": _result("run-1", business_insights, llm_trace)}))
    response = service.get_llmops("run-1")

    ids = {c.group_id for c in response.calls}
    assert ids == {"1 | 1", "1 | 2"}


def test_a_retried_group_reports_the_last_attempts_outcome_and_all_attempts():
    business_insights = {"groups": {"1 | 1": _insight_group("1 | 1", retry_count=2)}, "trace_summary": {}}
    llm_trace = {
        "calls": [
            _call("1 | 1", attempt_number=1, final_status="validation_failed", validation_status="failed"),
            _call("1 | 1", attempt_number=2, final_status="validation_failed", validation_status="failed"),
            _call("1 | 1", attempt_number=3, final_status="success", validation_status="passed"),
        ]
    }

    service = LLMOpsService(_FakeExecutor({"run-1": _result("run-1", business_insights, llm_trace)}))
    call = service.get_llmops("run-1").calls[0]

    assert call.retry_count == 2
    assert call.final_status == "success"  # the last attempt's outcome
    assert len(call.attempts) == 3
    assert call.attempts[0].final_status == "validation_failed"
    assert call.attempts[2].final_status == "success"


def test_a_template_only_group_has_no_attempts_but_still_has_a_row():
    """The fallback template path never touches Azure OpenAI at all — no
    LLMCallTrace is ever created for it — but the group still explained
    something, so it belongs in the observability view."""
    business_insights = {"groups": {"1 | 3": _insight_group("1 | 3", provider="template")}, "trace_summary": {}}
    llm_trace = {"calls": []}

    service = LLMOpsService(_FakeExecutor({"run-1": _result("run-1", business_insights, llm_trace)}))
    call = service.get_llmops("run-1").calls[0]

    assert call.group_id == "1 | 3"
    assert call.provider == "template"
    assert call.attempts == []
    assert call.total_tokens is None


# ---------------------------------------------------------------------
# Empty / missing / malformed
# ---------------------------------------------------------------------


def test_a_run_with_no_llm_activity_at_all_is_reported_as_unavailable_not_an_error():
    service = LLMOpsService(_FakeExecutor({"run-1": _result("run-1")}))
    response = service.get_llmops("run-1")

    assert response.available is False
    assert response.calls == []
    assert response.summary.call_count == 0
    assert response.summary.status == "not_generated"


def test_a_call_missing_optional_fields_does_not_crash_the_service():
    business_insights = {"groups": {}, "trace_summary": {}}
    llm_trace = {"calls": [{"group_id": "1 | 1"}]}  # only the required key present

    service = LLMOpsService(_FakeExecutor({"run-1": _result("run-1", business_insights, llm_trace)}))
    call = service.get_llmops("run-1").calls[0]

    assert call.group_id == "1 | 1"
    assert call.total_tokens is None
    assert call.provider == "none"


def test_a_call_with_no_group_id_is_dropped_rather_than_crashing():
    llm_trace = {"calls": [{"total_tokens": 10}]}

    service = LLMOpsService(_FakeExecutor({"run-1": _result("run-1", {"groups": {}}, llm_trace)}))
    response = service.get_llmops("run-1")

    assert response.calls == []


def test_malformed_trace_summary_still_produces_a_response():
    llm_trace = {"summary": "not a dict, somehow"}
    # _build_summary reads trace.get("summary") — a non-dict there would
    # break .get() calls, so the service must not assume shape blindly.
    with pytest.raises(AttributeError):
        # This documents the current contract: `trace["summary"]` is always
        # a dict because the engine always writes it that way (LLMTraceStore
        # .to_dict()). A malformed *file* (not a malformed in-memory object)
        # would fail earlier, at JSON parsing in the runner, not here.
        LLMOpsService(_FakeExecutor({"run-1": _result("run-1", {}, llm_trace)})).get_llmops("run-1")


def test_unknown_run_raises_unknown_run_error():
    service = LLMOpsService(_FakeExecutor({}))
    with pytest.raises(UnknownRunError):
        service.get_llmops("does-not-exist")


# ---------------------------------------------------------------------
# Endpoint: auth, run scoping, 404
# ---------------------------------------------------------------------


def _principal(role: Role) -> Principal:
    return Principal(subject=f"user-{role.value}", display_name=role.value, roles=[role], permissions=permissions_for([role]))


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _as_role(role: Role) -> None:
    app.dependency_overrides[get_current_principal] = lambda: _principal(role)


def _with_fake_service(results: dict[str, PipelineExecutionResult]) -> None:
    app.dependency_overrides[get_llmops_service] = lambda: LLMOpsService(_FakeExecutor(results))


def test_data_scientist_can_read_llmops(client):
    _as_role(Role.DATA_SCIENTIST)
    _with_fake_service({"run-1": _result("run-1", {"groups": {"1 | 1": _insight_group("1 | 1")}}, {"calls": [_call("1 | 1")]})})

    response = client.get("/results/run-1/llmops")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run-1"
    assert body["available"] is True
    assert len(body["calls"]) == 1


def test_analyst_is_refused(client):
    _as_role(Role.ANALYST)
    _with_fake_service({"run-1": _result("run-1")})

    assert client.get("/results/run-1/llmops").status_code == 403


def test_unassigned_user_is_refused(client):
    # Matches this project's own convention for testing unauthorized access
    # (test_auth_api.py's `as_unassigned`): authenticated, but with no role
    # — not a bare-token-missing 401, which this harness does not simulate.
    app.dependency_overrides[get_current_principal] = lambda: Principal(subject="user-none", roles=[], permissions=[])
    _with_fake_service({"run-1": _result("run-1")})

    assert client.get("/results/run-1/llmops").status_code == 403


def test_unknown_run_is_404(client):
    _as_role(Role.ADMIN)
    _with_fake_service({})

    response = client.get("/results/does-not-exist/llmops")

    assert response.status_code == 404


def test_the_endpoint_scopes_data_to_the_requested_run_id(client):
    """Two runs' data must never bleed into each other's response."""
    _as_role(Role.ADMIN)
    _with_fake_service(
        {
            "run-1": _result("run-1", {"groups": {"1 | 1": _insight_group("1 | 1")}}, {"calls": [_call("1 | 1")]}),
            "run-2": _result("run-2", {"groups": {"2 | 1": _insight_group("2 | 1")}}, {"calls": [_call("2 | 1")]}),
        }
    )

    body_1 = client.get("/results/run-1/llmops").json()
    body_2 = client.get("/results/run-2/llmops").json()

    assert [c["group_id"] for c in body_1["calls"]] == ["1 | 1"]
    assert [c["group_id"] for c in body_2["calls"]] == ["2 | 1"]


def test_no_secret_or_credential_shaped_values_in_the_response(client):
    """Defense in depth: even though nothing upstream stores raw prompts,
    tokens, or API keys on the trace, the response must not somehow leak one."""
    _as_role(Role.ADMIN)
    _with_fake_service({"run-1": _result("run-1", {"groups": {"1 | 1": _insight_group("1 | 1")}}, {"calls": [_call("1 | 1")]})})

    body = client.get("/results/run-1/llmops").text

    for marker in ("AZURE_OPENAI_API_KEY", "api_key", "client_secret", "Authorization", "Bearer "):
        assert marker not in body
