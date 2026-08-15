"""Extracts the numeric metrics MLflow records.

MLflow metrics are numeric only, so categorical results (validation status,
selection status) are logged as tags, with a companion 0/1 metric so they
stay queryable in the metrics view.

Keys are namespaced <section>.<group>.<model>.<name> so a run spanning many
groups and models never collides.
"""

from __future__ import annotations


from forecast_engine.core.pipeline_result import PipelineResult

_BACKTEST_METRIC_FIELDS = ("mape", "wmape", "rmse", "mae", "smape", "accuracy")


# Build every numeric run metric: training, backtest, ranking, drift, selection
def build_metrics(pipeline_result: PipelineResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics.update(_training_metrics(pipeline_result))
    metrics.update(_backtesting_metrics(pipeline_result))
    metrics.update(_validation_metrics(pipeline_result))
    metrics.update(_ranking_metrics(pipeline_result))
    metrics.update(_drift_and_threshold_metrics(pipeline_result))
    metrics.update(_selection_metrics(pipeline_result))
    return metrics


# Build categorical status tags for validation and final selection
def build_status_tags(pipeline_result: PipelineResult) -> dict[str, str]:
    tags: dict[str, str] = {}
    for result in pipeline_result.forward_validation_results.get("results", []):
        key = f"validation.{result['group_id']}.{result['model_name']}.status"
        tags[key] = result.get("status", "Unknown")

    for winner in pipeline_result.final_winner_models:
        key = f"selection.{winner['forecast_group']}.status"
        tags[key] = winner.get("final_selection_status", "Unknown")
    return tags


# Extract training count metrics
def _training_metrics(pipeline_result: PipelineResult) -> dict[str, float]:
    training = pipeline_result.training_summary
    if not training:
        return {}
    return {
        "training.trained": training.get("trained"),
        "training.failed": training.get("failed"),
        "training.skipped": training.get("skipped"),
        "training.unavailable": training.get("unavailable"),
    }


# Extract per (group, model) backtesting metrics
def _backtesting_metrics(pipeline_result: PipelineResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for result in pipeline_result.backtesting_metrics.get("results", []):
        prefix = f"backtest.{result['group_id']}.{result['model_name']}"
        overall = (result.get("backtest") or {}).get("overall") or {}
        for field_name in _BACKTEST_METRIC_FIELDS:
            value = overall.get(field_name)
            if value is not None:
                metrics[f"{prefix}.{field_name}"] = value
    return metrics


# Encode Forward Validation pass/fail as a 0/1 metric per (group, model)
def _validation_metrics(pipeline_result: PipelineResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for result in pipeline_result.forward_validation_results.get("results", []):
        key = f"validation.{result['group_id']}.{result['model_name']}.survived"
        metrics[key] = 1.0 if result.get("status") == "Survived" else 0.0
    return metrics


# Extract per (group, model) composite ranking score and rank
def _ranking_metrics(pipeline_result: PipelineResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for group_id, models in pipeline_result.ranking_results.get("rankings", {}).items():
        for model in models:
            prefix = f"ranking.{group_id}.{model['model_name']}"
            if model.get("composite_score") is not None:
                metrics[f"{prefix}.composite_score"] = model["composite_score"]
            if model.get("final_composite_rank") is not None:
                metrics[f"{prefix}.final_rank"] = model["final_composite_rank"]
    return metrics


# Extract per-group drift statistic and threshold value
def _drift_and_threshold_metrics(pipeline_result: PipelineResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for group_id, drift in pipeline_result.drift_results.items():
        if drift.get("statistic") is not None:
            metrics[f"drift.{group_id}.statistic"] = drift["statistic"]
    for group_id, threshold in pipeline_result.threshold_results.items():
        if threshold.get("value") is not None:
            metrics[f"threshold.{group_id}.value"] = threshold["value"]
    return metrics


# Encode Final Production Model Selection outcome as a 0/1 metric per group
def _selection_metrics(pipeline_result: PipelineResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for winner in pipeline_result.final_winner_models:
        key = f"selection.{winner['forecast_group']}.selected"
        metrics[key] = 1.0 if winner.get("final_selection_status") != "No Model Available" else 0.0
        if winner.get("composite_ranking_score") is not None:
            metrics[f"selection.{winner['forecast_group']}.composite_score"] = winner["composite_ranking_score"]
    return metrics
