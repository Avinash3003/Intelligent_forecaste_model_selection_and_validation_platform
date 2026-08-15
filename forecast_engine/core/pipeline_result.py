"""The finished run, as every downstream consumer sees it.

The LLM layer and the MLflow tracking layer both read exactly this object;
neither derives its own view from PipelineContext.

Building it is the only place allowed to reach into PipelineContext — past
this module nothing sees a live model, DataFrame or config object. It is
assembled from each stage's own to_dict(), the same plain-JSON contract the
API and run summary already use.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forecast_engine.core.pipeline_context import PipelineContext


@dataclass
class PipelineResult:
    """One run's finished record, entirely plain-JSON data."""

    run_id: str
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    forecast_configuration: dict[str, Any] = field(default_factory=dict)
    pipeline_configuration: dict[str, Any] = field(default_factory=dict)
    forecast_horizon: int | None = None
    curated_dataset_uri: str | None = None
    forecast_groups: list[dict[str, Any]] = field(default_factory=list)
    selected_models: list[str] = field(default_factory=list)
    # Model configured for the fallback path (Section 6.9). Whether it was
    # actually used is per-group, on `final_winner_models`.
    fallback_model: str | None = None
    hyperparameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    training_summary: dict[str, Any] = field(default_factory=dict)
    backtesting_metrics: dict[str, Any] = field(default_factory=dict)
    forward_validation_results: dict[str, Any] = field(default_factory=dict)
    explainability_results: dict[str, Any] = field(default_factory=dict)
    ranking_results: dict[str, Any] = field(default_factory=dict)
    drift_results: dict[str, Any] = field(default_factory=dict)
    threshold_results: dict[str, Any] = field(default_factory=dict)
    final_winner_models: list[dict[str, Any]] = field(default_factory=list)
    forecast_outputs: list[dict[str, Any]] = field(default_factory=list)
    business_insights: dict[str, Any] = field(default_factory=dict)
    # Detailed per-call LLM trace (Section 13.4) — separate from the
    # aggregate counts inside `business_insights["trace_summary"]`.
    llm_trace: dict[str, Any] = field(default_factory=dict)

    # Serialize for logging, MLflow tracking and the LLM Insight Engine
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_metadata": self.dataset_metadata,
            "forecast_configuration": self.forecast_configuration,
            "pipeline_configuration": self.pipeline_configuration,
            "forecast_horizon": self.forecast_horizon,
            "curated_dataset_uri": self.curated_dataset_uri,
            "forecast_groups": self.forecast_groups,
            "selected_models": self.selected_models,
            "fallback_model": self.fallback_model,
            "hyperparameters": self.hyperparameters,
            "training_summary": self.training_summary,
            "backtesting_metrics": self.backtesting_metrics,
            "forward_validation_results": self.forward_validation_results,
            "explainability_results": self.explainability_results,
            "ranking_results": self.ranking_results,
            "drift_results": self.drift_results,
            "threshold_results": self.threshold_results,
            "final_winner_models": self.final_winner_models,
            "forecast_outputs": self.forecast_outputs,
            "business_insights": self.business_insights,
            "llm_trace": self.llm_trace,
        }


class PipelineResultBuilder:
    """Builds a PipelineResult from a finished context — the one place allowed
    to read PipelineContext for LLM or MLflow purposes."""

    # Build a PipelineResult from a finished PipelineContext
    def build(self, context: "PipelineContext") -> PipelineResult:
        training = context.training_report.to_dict() if context.training_report else {}
        evaluation = context.evaluation_report.to_dict() if context.evaluation_report else {}
        explainability = context.explainability_report.to_dict() if context.explainability_report else {}
        ranking = context.ranking_report.to_dict() if context.ranking_report else {}
        selection = context.production_selection_report.to_dict() if context.production_selection_report else {}

        hyperparameters = {
            f"{result['group_id']} | {result['model_name']}": result["params"]
            for result in training.get("results", [])
            if result.get("status") == "Trained"
        }

        winners = selection.get("results", [])
        drift_results = {
            winner["forecast_group"]: {
                "algorithm": winner.get("selected_drift_algorithm"),
                "statistic": winner.get("drift_statistic"),
                "result": winner.get("drift_validation_result"),
            }
            for winner in winners
        }
        threshold_results = {
            winner["forecast_group"]: {
                "method": winner.get("dynamic_threshold_method"),
                "value": winner.get("dynamic_threshold_value"),
            }
            for winner in winners
        }
        forecast_outputs = [
            {
                "forecast_group": winner["forecast_group"],
                "model_name": winner.get("final_production_model"),
                "fallback_used": winner.get("fallback_flag"),
                "forecast": winner.get("forecast"),
            }
            for winner in winners
        ]

        return PipelineResult(
            run_id=context.run_id,
            dataset_metadata={
                "dataset_path": str(context.dataset_path),
                "frequency": context.frequency,
                "mode": context.mode.value,
                "raw_rows": context.metadata.get("raw_rows"),
                "raw_columns": context.metadata.get("raw_columns"),
                "group_count": context.group_count,
                "series_count": context.series_count,
            },
            forecast_configuration=context.configuration.to_dict(),
            pipeline_configuration=_jsonable_dataclass(context.pipeline_config),
            forecast_horizon=_first_forecast_horizon(evaluation),
            curated_dataset_uri=context.curated_dataset_uri,
            forecast_groups=[series.to_dict() for series in context.series],
            selected_models=context.selected_models or [],
            fallback_model=context.fallback_model,
            hyperparameters=hyperparameters,
            training_summary=training,
            backtesting_metrics=evaluation,
            forward_validation_results=evaluation,
            explainability_results=explainability,
            ranking_results=ranking,
            drift_results=drift_results,
            threshold_results=threshold_results,
            final_winner_models=winners,
            forecast_outputs=forecast_outputs,
            business_insights=context.insight_report.to_dict() if context.insight_report else {},
            llm_trace=context.llm_trace,
        )


# Serialize a (possibly nested) dataclass into plain JSON types
def _jsonable_dataclass(value: Any) -> Any:
    # PipelineConfig is never modified to grow a to_dict() of its own —
    # preprocessing configuration is out of this and every later phase's
    # scope — so this reads it purely externally, converting Enum members to
    # their .value along the way (dataclasses.asdict alone leaves them as
    # Enum instances, which are not JSON-serializable).
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable_dataclass(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_jsonable_dataclass(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable_dataclass(item) for key, item in value.items()}
    return value


# Derive the configured forecast horizon from any produced forecast
def _first_forecast_horizon(evaluation: dict[str, Any]) -> int | None:
    # Every model in a run is generated with the same configured horizon
    # (Section 6.5), so the first available one is representative — this
    # avoids the builder needing a direct reference to EvaluationConfig,
    # which lives on the pipeline orchestrator, not on PipelineContext.
    for result in evaluation.get("results", []):
        forecast = result.get("forecast")
        if forecast and forecast.get("horizon"):
            return forecast["horizon"]
    return None
