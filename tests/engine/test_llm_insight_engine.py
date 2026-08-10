"""End-to-end LLMOps behaviour of `LLMInsightEngine`: routing, structured
output validation with bounded retry, grounding blocking, provider
fallback, and per-run token budget enforcement.

Uses a fake `AzureOpenAIService` (duck-typed to the same interface) rather
than mocking the SDK — this exercises the engine's real orchestration logic
without needing network access or credentials.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

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
