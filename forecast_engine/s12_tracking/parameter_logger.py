"""Parameter extraction (Section 6.13, "Log Parameters").

Every parameter name is derived from the data itself — a business key
column, a model name — rather than a fixed literal, so a dataset with
different keys or a differently-named model produces differently-named
parameters without any code change here.
"""

from __future__ import annotations

from typing import Any

from forecast_engine.config.mlflow_config import MLflowConfig
from forecast_engine.core.pipeline_result import PipelineResult


# Build top-level run parameters (config, models, horizon, frequency)
def build_run_parameters(pipeline_result: PipelineResult) -> dict[str, Any]:
    cfg = pipeline_result.forecast_configuration
    meta = pipeline_result.dataset_metadata

    return {
        "run_id": pipeline_result.run_id,
        "date_column": cfg.get("date_column"),
        "target_column": cfg.get("target_column"),
        "key_columns": ",".join(cfg.get("key_columns", [])) or "(single series)",
        "feature_columns": ",".join(cfg.get("feature_columns", [])) or "(none)",
        "mode": cfg.get("mode"),
        "aggregation_method": cfg.get("aggregation_method"),
        "forecast_horizon": pipeline_result.forecast_horizon,
        "dataset_frequency": meta.get("frequency"),
        "selected_models": ",".join(pipeline_result.selected_models) or "(all registered models)",
        "fallback_model": pipeline_result.fallback_model or "(configured default)",
        "fallback_triggered_groups": _fallback_triggered_count(pipeline_result),
        "forecast_group_count": meta.get("group_count"),
        "series_count": meta.get("series_count"),
    }


# Count groups that fell back after every evaluated model failed
def _fallback_triggered_count(pipeline_result: PipelineResult) -> int:
    return sum(1 for winner in pipeline_result.final_winner_models if winner.get("fallback_flag"))


# Flatten per-model hyperparameters, capped at config.max_hyperparameter_params
def build_hyperparameters(pipeline_result: PipelineResult, config: MLflowConfig) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, params in pipeline_result.hyperparameters.items():
        # Full unbounded set is always available via the "hyperparameters" JSON artifact
        if len(flattened) >= config.max_hyperparameter_params:
            break
        for param_name, value in params.items():
            flattened[f"hp.{key}.{param_name}"] = value
    return flattened
