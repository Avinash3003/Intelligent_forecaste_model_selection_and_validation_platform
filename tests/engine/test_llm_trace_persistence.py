"""The detailed per-call LLM trace reaching MLflow.

`LLMInsightEngine.generate()` builds a full `LLMCallTrace` per attempt, but
until now nothing carried it past the run — only the aggregate
`trace_summary` (call count, totals) survived into `summary.json`, and the
per-call detail (which group, which deployment, what validation/grounding
result, per-call tokens) was built and then discarded. These cover the
three hops that now carry it through: PipelineContext -> PipelineResult ->
the MLflow artifact logger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forecast_engine.config.mlflow_config import MLflowConfig
from forecast_engine.core.forecast_configuration import ForecastConfiguration
from forecast_engine.core.pipeline_context import PipelineContext
from forecast_engine.core.pipeline_result import PipelineResult, PipelineResultBuilder
from forecast_engine.s12_tracking.artifact_logger import _log_llm_business_summary

_TRACE = {
    "run_id": "run-1",
    "summary": {"call_count": 1, "total_tokens": 50},
    "calls": [{"group_id": "1 | 1", "final_status": "success"}],
}


def _context(llm_trace: dict[str, Any]) -> PipelineContext:
    ctx = PipelineContext(
        run_id="run-1",
        dataset_path="x.csv",
        configuration=ForecastConfiguration(date_column="date", target_column="sales"),
    )
    ctx.llm_trace = llm_trace
    return ctx


def test_the_trace_is_included_in_the_context_summary():
    ctx = _context(_TRACE)

    assert ctx.summary()["llm_trace"] == _TRACE


def test_the_builder_carries_the_trace_from_context_to_the_result():
    ctx = _context(_TRACE)

    result = PipelineResultBuilder().build(ctx)

    assert result.llm_trace == _TRACE
    assert result.to_dict()["llm_trace"] == _TRACE


def test_a_run_with_no_llm_activity_carries_an_empty_trace():
    ctx = _context({})

    assert PipelineResultBuilder().build(ctx).llm_trace == {}


@dataclass
class _FakeMLflowClient:
    logged_dicts: list = field(default_factory=list)

    def log_dict_artifact(self, data, path):
        self.logged_dicts.append((path, data))

    def log_text_artifact(self, text, path):
        pass


def _result(llm_trace: dict[str, Any], available: bool = False) -> PipelineResult:
    result = PipelineResult(run_id="run-1")
    result.business_insights = {"available": available}
    result.llm_trace = llm_trace
    return result


def test_the_trace_is_logged_as_its_own_mlflow_artifact():
    client = _FakeMLflowClient()

    _log_llm_business_summary(client, _result(_TRACE), MLflowConfig())

    paths = [path for path, _ in client.logged_dicts]
    assert "insights/llm_trace.json" in paths


def test_a_run_with_no_calls_logs_no_trace_artifact():
    client = _FakeMLflowClient()

    _log_llm_business_summary(client, _result({}), MLflowConfig())

    paths = [path for path, _ in client.logged_dicts]
    assert "insights/llm_trace.json" not in paths


def test_the_trace_is_logged_even_when_every_call_failed():
    """The exact case this exists to debug: insights unavailable, but the
    attempt(s) that failed are still on record."""
    client = _FakeMLflowClient()
    failed_trace = {"run_id": "run-1", "summary": {"call_count": 1}, "calls": [{"final_status": "provider_error"}]}

    _log_llm_business_summary(client, _result(failed_trace, available=False), MLflowConfig())

    paths = [path for path, _ in client.logged_dicts]
    assert "insights/llm_trace.json" in paths
