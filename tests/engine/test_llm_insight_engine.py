"""End-to-end LLMOps behaviour of `LLMInsightEngine`: routing, structured
output validation with bounded retry, grounding blocking, provider
fallback, and per-run token budget enforcement.

Uses a fake `AzureOpenAIService` (duck-typed to the same interface) rather
than mocking the SDK — this exercises the engine's real orchestration logic
without needing network access or credentials.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass

import pytest

from forecast_engine.config.llm_config import LLMConfig
from forecast_engine.core.pipeline_result import PipelineResult
from forecast_engine.s11_llm.azure_openai_service import LLMCompletionResult
from forecast_engine.s11_llm.insight_engine import LLMInsightEngine
from forecast_engine.utils.exceptions import LLMProviderError


def _valid_json(model: str, wmape: float) -> str:
    return json.dumps(
        {
            "selected_model": model,
            "rejection_reasons": [],
            "confidence": round(100 - wmape, 1),
            "caveats": [],
            "concise_summary": f"{model} was selected, with a backtest WMAPE of {wmape}%, off by about "
            f"{round(wmape)}% on average.",
        }
    )


@dataclass
class _Call:
    system_prompt: str
    user_prompt: str
    deployment: str | None
    use_fallback: bool
    json_mode: bool


class FakeAzureOpenAIService:
    """Scripted `AzureOpenAIService` stand-in.

    `responses` is consumed one item per call, in order: either a string
    (the response text, tokens defaulted) or an Exception to raise. Running
    out of scripted responses raises AssertionError — a test controls
    exactly how many calls it expects.
    """

    def __init__(
        self,
        responses: list,
        *,
        primary_available: bool = True,
        fallback_available: bool = False,
        tokens_per_call: tuple[int, int] = (500, 100),
    ) -> None:
        self._responses = list(responses)
        self._primary_available = primary_available
        self._fallback_available = fallback_available
        self._tokens_per_call = tokens_per_call
        self.calls: list[_Call] = []

    def is_available(self, *, use_fallback: bool = False) -> bool:
        return self._fallback_available if use_fallback else self._primary_available

    def unavailable_reason(self, *, use_fallback: bool = False) -> str:
        return "not configured"

    def complete(self, system_prompt, user_prompt, *, deployment=None, use_fallback=False, json_mode=False, max_tokens=None):
        self.calls.append(_Call(system_prompt, user_prompt, deployment, use_fallback, json_mode))
        if not self._responses:
            raise AssertionError("FakeAzureOpenAIService ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        prompt_tokens, completion_tokens = self._tokens_per_call
        return LLMCompletionResult(
            text=item, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens, latency_ms=12.5,
        )


def _make_result(groups: list[dict]) -> PipelineResult:
    """Build a minimal PipelineResult with the fields
    `context_formatter.group_decision_facts`/`group_grounding_metrics`
    read, for however many groups the test needs.

    Each `groups` entry: {"group_id", "model", "wmape", "rejected": [...]}
    """
    final_winner_models = []
    backtest_results = []
    forecast_groups = []

    for g in groups:
        forecast_groups.append({"group_id": g["group_id"], "meets_minimum_history": True})
        final_winner_models.append(
            {
                "forecast_group": g["group_id"],
                "final_production_model": g["model"],
                "fallback_flag": g.get("is_fallback", False),
                "fallback_trigger": g.get("fallback_trigger"),
                "rejected_candidates": g.get("rejected", []),
                "failure_reasons": g.get("failure_reasons", []),
            }
        )
        if g.get("wmape") is not None:
            backtest_results.append(
                {
                    "group_id": g["group_id"],
                    "model_name": g["model"],
                    "status": "Survived",
                    "backtest": {"overall": {"wmape": g["wmape"], "accuracy": 100 - g["wmape"]}},
                }
            )

    return PipelineResult(
        run_id="fe-run-test",
        dataset_metadata={"dataset_path": "x.csv", "raw_rows": 100, "raw_columns": 3, "frequency": "Monthly", "mode": "Multi Series", "group_count": len(groups), "series_count": len(groups)},
        forecast_configuration={"date_column": "date", "target_column": "sales", "key_columns": ["store"], "feature_columns": []},
        forecast_groups=forecast_groups,
        selected_models=["xgboost", "lightgbm"],
        fallback_model="seasonal_naive",
        backtesting_metrics={"results": backtest_results},
        forward_validation_results={"results": []},
        explainability_results={"results": []},
        ranking_results={"rankings": {}},
        drift_results={},
        threshold_results={},
        final_winner_models=final_winner_models,
    )


@pytest.fixture
def config():
    return LLMConfig(
        enabled=True, endpoint="https://example.openai.azure.com/", api_key="k",
        deployment_name="gpt-primary", deployment_name_simple="gpt-cheap", deployment_name_complex="gpt-strong",
        max_validation_retries=2,
        price_input_per_1k=0.15, price_output_per_1k=0.6,
    )


def test_happy_path_returns_llm_generated_grounded_payload(config):
    result = _make_result([{"group_id": "1 | 1", "model": "xgboost", "wmape": 8.2}])
    service = FakeAzureOpenAIService([_valid_json("xgboost", 8.2)])
    engine = LLMInsightEngine(config=config, service=service)

    report = engine.generate(result)

    insight = report.groups["1 | 1"]
    assert insight.provider == "azure_openai"
    assert insight.validation_status == "passed"
    assert insight.grounding_status == "grounded"
    assert insight.retry_count == 0
    assert insight.payload.selected_model == "xgboost"
    assert report.trace_summary["call_count"] == 1
    assert report.trace_summary["total_tokens"] == 600
    assert report.trace_summary["estimated_cost_usd"] == pytest.approx(0.15 * 0.5 + 0.6 * 0.1)


def test_the_detailed_per_call_trace_is_available_after_generate(config):
    """`trace_summary` on the report is aggregate-only; Section 13.4's
    actual per-call record (one entry per attempt, with its own tokens,
    validation, and grounding status) must be reachable separately so it
    can be persisted, not just counted."""
    result = _make_result([{"group_id": "1 | 1", "model": "xgboost", "wmape": 8.2}])
    service = FakeAzureOpenAIService([_valid_json("xgboost", 8.2)])
    engine = LLMInsightEngine(config=config, service=service)

    engine.generate(result)

    store = engine.trace_store
    assert store is not None
    calls = store.to_dict()["calls"]
    assert len(calls) == 1
    assert calls[0]["group_id"] == "1 | 1"
    assert calls[0]["validation_status"] == "passed"
    assert calls[0]["grounding_status"] == "grounded"
    assert calls[0]["final_status"] == "success"


def test_trace_store_is_none_before_generate_has_run(config):
    engine = LLMInsightEngine(config=config, service=FakeAzureOpenAIService([]))

    assert engine.trace_store is None


def test_routing_sends_single_rejection_group_to_the_simple_deployment(config):
    result = _make_result([{"group_id": "1 | 1", "model": "xgboost", "wmape": 8.2, "rejected": []}])
    service = FakeAzureOpenAIService([_valid_json("xgboost", 8.2)])
    LLMInsightEngine(config=config, service=service).generate(result)

    assert service.calls[0].deployment == "gpt-cheap"


def test_routing_sends_multi_rejection_group_to_the_complex_deployment(config):
    rejected = [{"model_name": "prophet", "reason": "x"}, {"model_name": "arima", "reason": "y"}]
    result = _make_result([{"group_id": "1 | 1", "model": "xgboost", "wmape": 8.2, "rejected": rejected}])
    service = FakeAzureOpenAIService([_valid_json("xgboost", 8.2)])
    LLMInsightEngine(config=config, service=service).generate(result)

    assert service.calls[0].deployment == "gpt-strong"


def test_invalid_json_triggers_a_retry_with_the_failure_fed_back(config):
    result = _make_result([{"group_id": "1 | 1", "model": "xgboost", "wmape": 8.2}])
    service = FakeAzureOpenAIService(["not json at all", _valid_json("xgboost", 8.2)])
    engine = LLMInsightEngine(config=config, service=service)

    report = engine.generate(result)

    insight = report.groups["1 | 1"]
    assert insight.provider == "azure_openai"
    assert insight.retry_count == 1
    assert len(service.calls) == 2
    # The retry prompt embeds the validation failure, per Section 13.1.
    assert "failed validation" in service.calls[1].user_prompt


def test_a_hallucinated_number_is_blocked_and_retried(config):
    # 55% matches neither the real WMAPE (8.2) nor its accuracy complement
    # (91.8, also a legitimate fact this group carries) within tolerance —
    # a genuine fabrication, not a rounding of a real number.
    hallucinated = json.dumps(
        {"selected_model": "xgboost", "rejection_reasons": [], "confidence": 8.2,
         "caveats": [], "concise_summary": "xgboost had a WMAPE of 55.0%, a mediocre result."}
    )
    result = _make_result([{"group_id": "1 | 1", "model": "xgboost", "wmape": 8.2}])
    service = FakeAzureOpenAIService([hallucinated, _valid_json("xgboost", 8.2)])
    engine = LLMInsightEngine(config=config, service=service)

    report = engine.generate(result)

    insight = report.groups["1 | 1"]
    assert insight.provider == "azure_openai"  # eventually succeeded, on the LLM path
    assert insight.grounding_status == "grounded"
    assert insight.retry_count == 1
    assert "cited number(s)" in service.calls[1].user_prompt


def test_exhausted_retries_fall_through_to_the_template(config):
    config = LLMConfig(
        enabled=True, endpoint="https://x", api_key="k", deployment_name="gpt-primary",
        max_validation_retries=0,
    )
    result = _make_result([{"group_id": "1 | 1", "model": "xgboost", "wmape": 8.2}])
    service = FakeAzureOpenAIService(["still not json"])
    engine = LLMInsightEngine(config=config, service=service)

    report = engine.generate(result)

    insight = report.groups["1 | 1"]
    assert insight.provider == "template"
    assert insight.payload is not None  # the run still finishes with an explanation
    assert len(service.calls) == 1  # no retries were attempted (max_validation_retries=0)


def test_primary_unavailable_uses_the_fallback_deployment(config):
    config = LLMConfig(
        enabled=True, endpoint="https://x", api_key="k", deployment_name="gpt-primary",
        fallback_endpoint="https://backup", fallback_api_key="k2", fallback_deployment_name="gpt-backup",
    )
    result = _make_result([{"group_id": "1 | 1", "model": "xgboost", "wmape": 8.2}])
    service = FakeAzureOpenAIService(
        [_valid_json("xgboost", 8.2)], primary_available=False, fallback_available=True,
    )
    engine = LLMInsightEngine(config=config, service=service)

    report = engine.generate(result)

    insight = report.groups["1 | 1"]
    assert insight.provider == "azure_openai_fallback"
    assert service.calls[0].deployment == "gpt-backup"
    assert service.calls[0].use_fallback is True


def test_no_provider_at_all_uses_the_template_and_still_finishes(config):
    result = _make_result([{"group_id": "1 | 1", "model": "xgboost", "wmape": 8.2}])
    service = FakeAzureOpenAIService([], primary_available=False, fallback_available=False)
    engine = LLMInsightEngine(config=config, service=service)

    report = engine.generate(result)

    insight = report.groups["1 | 1"]
    assert insight.provider == "template"
    assert insight.payload is not None
    assert len(service.calls) == 0


def test_a_provider_error_moves_to_the_next_provider_without_retrying_the_same_one(config):
    config = LLMConfig(
        enabled=True, endpoint="https://x", api_key="k", deployment_name="gpt-primary",
        fallback_endpoint="https://backup", fallback_api_key="k2", fallback_deployment_name="gpt-backup",
        max_validation_retries=2,
    )
    result = _make_result([{"group_id": "1 | 1", "model": "xgboost", "wmape": 8.2}])
    service = FakeAzureOpenAIService(
        [LLMProviderError("rate limited"), _valid_json("xgboost", 8.2)],
        primary_available=True, fallback_available=True,
    )
    engine = LLMInsightEngine(config=config, service=service)

    report = engine.generate(result)

    insight = report.groups["1 | 1"]
    assert insight.provider == "azure_openai_fallback"
    # One call to the primary (which errored, no retry) + one to the fallback.
    assert len(service.calls) == 2
    assert service.calls[0].use_fallback is False
    assert service.calls[1].use_fallback is True


def test_token_budget_exhausted_mid_run_falls_back_to_template_for_remaining_groups(config):
    config = LLMConfig(
        enabled=True, endpoint="https://x", api_key="k", deployment_name="gpt-primary",
        max_tokens_per_run=700,  # one call (~600 tokens) fits; a second would not
    )
    result = _make_result(
        [
            {"group_id": "1 | 1", "model": "xgboost", "wmape": 8.2},
            {"group_id": "1 | 2", "model": "lightgbm", "wmape": 15.5},
        ]
    )
    service = FakeAzureOpenAIService([_valid_json("xgboost", 8.2)], tokens_per_call=(500, 100))
    engine = LLMInsightEngine(config=config, service=service)

    report = engine.generate(result)

    assert report.groups["1 | 1"].provider == "azure_openai"
    assert report.groups["1 | 2"].provider == "template"
    assert report.token_budget_exhausted is True
    assert len(service.calls) == 1  # the second group never reached the LLM at all


def test_fallback_group_facts_are_used_when_llm_is_unavailable(config):
    result = _make_result(
        [
            {
                "group_id": "1 | 4", "model": "seasonal_naive", "wmape": None, "is_fallback": True,
                "fallback_trigger": "All 1 evaluated model(s) failed validation (xgboost).",
                "failure_reasons": [{"model_name": "xgboost", "reason": "drift statistic exceeded threshold"}],
            }
        ]
    )
    service = FakeAzureOpenAIService([], primary_available=False)
    engine = LLMInsightEngine(config=config, service=service)

    report = engine.generate(result)

    insight = report.groups["1 | 4"]
    assert insight.payload.selected_model == "seasonal_naive"
    assert "fallback" in insight.payload.concise_summary.lower()
    assert "fallback model used" in insight.payload.caveats


def test_llm_disabled_produces_no_calls_and_reports_disabled():
    config = LLMConfig(enabled=False)
    result = _make_result([{"group_id": "1 | 1", "model": "xgboost", "wmape": 8.2}])
    service = FakeAzureOpenAIService([])
    report = LLMInsightEngine(config=config, service=service).generate(result)

    assert report.status == "disabled"
    assert report.groups == {}
    assert len(service.calls) == 0


class _SlowConcurrentService:
    """Records how many calls overlap, so a test can tell real concurrency
    from a fast sequential loop.

    Every call answers correctly for whichever group it was asked about, so
    the engine's own ordering — not the response script — decides which
    insight lands where.
    """

    def __init__(self, delay_seconds: float = 0.05) -> None:
        self._delay = delay_seconds
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak_in_flight = 0
        self.call_count = 0

    def is_available(self, *, use_fallback: bool = False) -> bool:
        return not use_fallback

    def unavailable_reason(self, *, use_fallback: bool = False) -> str:
        return "not configured"

    def complete(self, system_prompt, user_prompt, *, deployment=None, use_fallback=False, json_mode=False, max_tokens=None):
        with self._lock:
            self.call_count += 1
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            time.sleep(self._delay)
        finally:
            with self._lock:
                self._in_flight -= 1
        model, wmape = _model_and_wmape_in(user_prompt)
        return LLMCompletionResult(
            text=_valid_json(model, wmape), prompt_tokens=500, completion_tokens=100,
            total_tokens=600, latency_ms=12.5,
        )


def _model_and_wmape_in(user_prompt: str) -> tuple[str, float]:
    """The winning model and WMAPE this prompt's own group selected, read
    back off the rendered context so a scripted answer is grounded in the
    group it actually answers for — not in whichever model the run happens
    to name first."""
    model = re.search(r"^model=(\S+?),", user_prompt, re.MULTILINE)
    wmape = re.search(r"wmape=([\d.]+)", user_prompt)
    if not model or not wmape:
        raise AssertionError(f"No selected model in the prompt: {user_prompt[:400]}")
    return model.group(1), float(wmape.group(1))


def _four_groups() -> PipelineResult:
    return _make_result(
        [
            {"group_id": "1 | 1", "model": "xgboost", "wmape": 8.2},
            {"group_id": "1 | 2", "model": "lightgbm", "wmape": 15.5},
            {"group_id": "1 | 3", "model": "prophet", "wmape": 11.0},
            {"group_id": "1 | 4", "model": "arima", "wmape": 6.4},
        ]
    )


def test_groups_are_generated_concurrently_not_one_after_another():
    """Insight generation is network wait, so a run's LLM time should scale
    with key count over the worker count, not with key count."""
    config = LLMConfig(
        enabled=True, endpoint="https://x", api_key="k", deployment_name="gpt-primary",
        insight_max_workers=4,
    )
    service = _SlowConcurrentService()

    report = LLMInsightEngine(config=config, service=service).generate(_four_groups())

    assert service.call_count == 4
    assert service.peak_in_flight > 1
    assert report.status == "generated"


def test_concurrent_generation_still_reports_groups_in_pipeline_order():
    """Insights are keyed by group, and two runs of the same data must list
    them the same way — completion order must not leak into the report."""
    config = LLMConfig(
        enabled=True, endpoint="https://x", api_key="k", deployment_name="gpt-primary",
        insight_max_workers=4,
    )
    result = _four_groups()

    report = LLMInsightEngine(config=config, service=_SlowConcurrentService()).generate(result)

    assert list(report.groups) == ["1 | 1", "1 | 2", "1 | 3", "1 | 4"]
    assert [i.payload.selected_model for i in report.groups.values()] == [
        "xgboost", "lightgbm", "prophet", "arima",
    ]


def test_worker_count_bounds_how_many_calls_are_in_flight():
    """The ceiling exists so a large run cannot outrun the deployment's
    tokens-per-minute quota."""
    config = LLMConfig(
        enabled=True, endpoint="https://x", api_key="k", deployment_name="gpt-primary",
        insight_max_workers=2,
    )
    service = _SlowConcurrentService()

    LLMInsightEngine(config=config, service=service).generate(_four_groups())

    assert service.peak_in_flight <= 2


def test_a_configured_token_ceiling_keeps_generation_sequential():
    """The ceiling is counted against tokens actually spent, which cannot be
    known while calls are still in flight — so a budgeted run gives up the
    concurrency rather than the enforcement."""
    config = LLMConfig(
        enabled=True, endpoint="https://x", api_key="k", deployment_name="gpt-primary",
        insight_max_workers=4, max_tokens_per_run=100_000,
    )
    service = _SlowConcurrentService()

    LLMInsightEngine(config=config, service=service).generate(_four_groups())

    assert service.call_count == 4
    assert service.peak_in_flight == 1
