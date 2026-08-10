"""LLM trace store (Section 13.4)."""

from forecast_engine.s11_llm.llm_trace import LLMTraceStore


def test_empty_store_summarizes_to_zeros_not_none_crashes():
    summary = LLMTraceStore("run-1").summary()
    assert summary["call_count"] == 0
    assert summary["groundedness_rate"] is None
    assert summary["estimated_cost_usd"] is None


def test_one_successful_call_is_captured_in_full():
    store = LLMTraceStore("run-1")
    trace = store.start_call(
        group_id="1 | 1", model_name="xgboost", prompt_version="v2", deployment="gpt-4o-mini", routing_tier="simple",
    )
    trace.prompt_tokens = 400
    trace.completion_tokens = 80
    trace.total_tokens = 480
    trace.estimated_cost_usd = 0.0012
    trace.validation_status = "passed"
    trace.grounding_status = "grounded"
    trace.final_status = "success"
    store.finish_call(trace)

    summary = store.summary()
    assert summary["call_count"] == 1
    assert summary["total_tokens"] == 480
    assert summary["grounded_count"] == 1
    assert summary["groundedness_rate"] == 1.0
    assert summary["average_latency_ms"] is not None
    assert summary["estimated_cost_usd"] == 0.0012


def test_retries_are_counted_by_attempt_number():
    store = LLMTraceStore("run-1")
    for attempt in (1, 2, 3):
        trace = store.start_call(
            group_id="1 | 1", model_name="xgboost", prompt_version="v2", deployment="d", routing_tier="simple",
            attempt_number=attempt,
        )
        store.finish_call(trace)

    assert store.summary()["retry_count"] == 2  # attempts 2 and 3 are retries


def test_cost_is_unavailable_when_any_call_has_no_cost():
    store = LLMTraceStore("run-1")
    priced = store.start_call(group_id="a", model_name="m", prompt_version="v2", deployment="d", routing_tier="simple")
    priced.total_tokens = 100
    priced.estimated_cost_usd = 0.01
    store.finish_call(priced)

    unpriced = store.start_call(group_id="b", model_name="m", prompt_version="v2", deployment="d", routing_tier="simple")
    unpriced.total_tokens = 100
    unpriced.estimated_cost_usd = None
    store.finish_call(unpriced)

    summary = store.summary()
    assert summary["cost_available"] is False
    assert summary["estimated_cost_usd"] is None


def test_final_statuses_are_tallied():
    store = LLMTraceStore("run-1")
    for status in ("success", "success", "validation_failed", "provider_error"):
        trace = store.start_call(group_id="a", model_name="m", prompt_version="v2", deployment="d", routing_tier="simple")
        trace.final_status = status
        store.finish_call(trace)

    assert store.summary()["final_statuses"] == {"success": 2, "validation_failed": 1, "provider_error": 1}


def test_to_dict_includes_every_call():
    store = LLMTraceStore("run-1")
    store.finish_call(store.start_call(group_id="a", model_name="m", prompt_version="v2", deployment="d", routing_tier="simple"))
    store.finish_call(store.start_call(group_id="b", model_name="m", prompt_version="v2", deployment="d", routing_tier="simple"))

    payload = store.to_dict()
    assert payload["run_id"] == "run-1"
    assert len(payload["calls"]) == 2
