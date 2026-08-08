"""Renders a `PipelineResult` into the plain-text context block every
prompt shares (Section 6.12 — "the LLM should receive the complete
pipeline context... understand the complete execution instead of isolated
metrics").

Kept separate from `PipelineResult` itself (a pure data object) and from
`PromptLibrary` (which only knows how to load/fill template files) so each
piece has exactly one job: this one turns structured data into the bounded
text every prompt is built around.
"""

from __future__ import annotations

from typing import Any

from forecast_engine.config.llm_config import LLMConfig
from forecast_engine.core.pipeline_result import PipelineResult


# Render the shared context block, truncated per config's limits
def render_context(result: PipelineResult, config: LLMConfig) -> str:
    sections = [
        _dataset_section(result),
        _groups_section(result, config),
        _training_section(result),
        _backtesting_section(result, config),
        _forward_validation_section(result, config),
        _explainability_section(result, config),
        _ranking_section(result, config),
        _drift_section(result, config),
        _winners_section(result, config),
        _fallback_section(result, config),
    ]
    return "\n\n".join(section for section in sections if section)


# Format the dataset & forecast configuration section
def _dataset_section(result: PipelineResult) -> str:
    meta = result.dataset_metadata
    cfg = result.forecast_configuration
    lines = [
        "## Dataset & Forecast Configuration",
        f"Run ID: {result.run_id}",
        f"Dataset: {meta.get('dataset_path')} ({meta.get('raw_rows')} rows, {meta.get('raw_columns')} columns)",
        f"Detected frequency: {meta.get('frequency')}; mode: {meta.get('mode')}",
        f"Date column: {cfg.get('date_column')}; Target column: {cfg.get('target_column')}",
        f"Key columns: {cfg.get('key_columns')}; Feature columns: {cfg.get('feature_columns')}",
        f"Forecast groups: {meta.get('group_count')}; forecast-ready series: {meta.get('series_count')}",
        f"Models requested: {', '.join(result.selected_models) if result.selected_models else 'all registered models'}",
    ]
    return "\n".join(lines)


# Format the forecast groups section
def _groups_section(result: PipelineResult, config: LLMConfig) -> str:
    if not result.forecast_groups:
        return ""
    lines = ["## Forecast Groups"]
    for group in result.forecast_groups[: config.max_groups_in_prompt]:
        lines.append(
            f"- {group.get('group_id')}: {group.get('observation_count')} observation(s), "
            f"{group.get('start_date')} to {group.get('end_date')}"
        )
    remaining = len(result.forecast_groups) - config.max_groups_in_prompt
    if remaining > 0:
        lines.append(f"- ... and {remaining} more group(s) not shown here.")
    return "\n".join(lines)


# Format the model training summary section
def _training_section(result: PipelineResult) -> str:
    training = result.training_summary
    if not training:
        return ""
    lines = [
        "## Model Training Summary",
        f"Trained: {training.get('trained')}, Failed: {training.get('failed')}, "
        f"Skipped: {training.get('skipped')}, Unavailable: {training.get('unavailable')}",
    ]
    if result.hyperparameters:
        lines.append("Selected hyperparameters (sample):")
        for key, params in list(result.hyperparameters.items())[:10]:
            lines.append(f"  - {key}: {params}")
    return "\n".join(lines)


# Format the backtesting metrics section
def _backtesting_section(result: PipelineResult, config: LLMConfig) -> str:
    results = result.backtesting_metrics.get("results", [])
    if not results:
        return ""
    lines = ["## Backtesting Metrics (Section 6.4)"]
    for item in results[: config.max_groups_in_prompt * 3]:
        backtest = item.get("backtest") or {}
        overall = backtest.get("overall") or {}
        lines.append(
            f"- {item.get('group_id')} / {item.get('model_name')}: status={item.get('status')}, "
            f"accuracy={overall.get('accuracy')}, wmape={overall.get('wmape')}, rmse={overall.get('rmse')}"
        )
    return "\n".join(lines)


# Format the forward forecast validation section
def _forward_validation_section(result: PipelineResult, config: LLMConfig) -> str:
    results = result.forward_validation_results.get("results", [])
    if not results:
        return ""
    lines = ["## Forward Forecast Validation (Section 6.5)"]
    for item in results[: config.max_groups_in_prompt * 3]:
        reasons = item.get("rejection_reasons") or []
        lines.append(
            f"- {item.get('group_id')} / {item.get('model_name')}: {item.get('status')}"
            + (f" — rejected for: {', '.join(reasons)}" if reasons else "")
        )
    return "\n".join(lines)


# Format the SHAP / feature importance section
def _explainability_section(result: PipelineResult, config: LLMConfig) -> str:
    results = result.explainability_results.get("results", [])
    if not results:
        return ""
    lines = ["## SHAP / Feature Importance (Section 6.10)"]
    for item in results[: config.max_groups_in_prompt * 3]:
        stability = item.get("stability_metrics") or {}
        top_features = sorted(
            (item.get("global_importance") or {}).items(), key=lambda pair: pair[1], reverse=True
        )[: config.max_important_features]
        lines.append(
            f"- {item.get('group_id')} / {item.get('model_name')}: method={item.get('method')}, "
            f"dominant_feature={stability.get('dominant_feature')}, "
            f"dominant_feature_consistency={stability.get('dominant_feature_consistency')}, "
            f"window_stability={stability.get('window_stability')}, "
            f"horizon_stability={stability.get('horizon_stability')}, "
            f"top_features={top_features}"
        )
    return "\n".join(lines)


# Format the model ranking section
def _ranking_section(result: PipelineResult, config: LLMConfig) -> str:
    rankings = result.ranking_results.get("rankings", {})
    if not rankings:
        return ""
    lines = ["## Model Ranking (Section 6.6)"]
    for group_id, models in list(rankings.items())[: config.max_groups_in_prompt]:
        lines.append(f"- {group_id}:")
        for model in models:
            lines.append(
                f"    {model.get('final_composite_rank')}. {model.get('model_name')} "
                f"(composite={model.get('composite_score')}, backtest_rank={model.get('original_backtest_rank')}, "
                f"backtest_score={model.get('backtest', {}).get('score')}, "
                f"stability_score={model.get('stability', {}).get('score')}, "
                f"shap_score={model.get('shap', {}).get('score')})"
            )
    return "\n".join(lines)


# Format the drift detection & threshold estimation section
def _drift_section(result: PipelineResult, config: LLMConfig) -> str:
    if not result.drift_results:
        return ""
    lines = ["## Drift Detection & Threshold Estimation (Sections 6.7-6.9)"]
    for group_id, drift in list(result.drift_results.items())[: config.max_groups_in_prompt]:
        threshold = result.threshold_results.get(group_id, {})
        drift_result = drift.get("result") or {}
        lines.append(
            f"- {group_id}: algorithm={drift.get('algorithm')}, statistic={drift.get('statistic')}, "
            f"threshold_method={threshold.get('method')}, threshold_value={threshold.get('value')}, "
            f"passed={drift_result.get('passed')}"
        )
    return "\n".join(lines)


# Format the final production model selection section
def _winners_section(result: PipelineResult, config: LLMConfig) -> str:
    if not result.final_winner_models:
        return ""
    lines = ["## Final Production Model Selection (Section 6.9)"]
    for winner in result.final_winner_models[: config.max_groups_in_prompt]:
        rejected = winner.get("rejected_candidates") or []
        shown = rejected[: config.max_rejected_models_in_prompt]
        lines.append(
            f"- {winner.get('forecast_group')}: model={winner.get('final_production_model')}, "
            f"status={winner.get('final_selection_status')}, fallback={winner.get('fallback_flag')}, "
            f"composite_score={winner.get('composite_ranking_score')}, "
            f"final_rank={winner.get('final_rank')}"
        )
        if shown:
            lines.append(f"    Rejected candidates: {[(c.get('model_name'), c.get('reason')) for c in shown]}")
    return "\n".join(lines)


# Format the fallback model usage section
def _fallback_section(result: PipelineResult, config: LLMConfig) -> str:
    # Grounds the narrative's fallback reasoning: why every model failed,
    # and which fallback took over. Omitted entirely when no group fell back.
    triggered = [winner for winner in result.final_winner_models if winner.get("fallback_flag")]
    if not triggered:
        return ""

    lines = [
        "## Fallback Model Usage (Section 6.9)",
        f"Configured fallback model: {result.fallback_model}",
        f"Groups where every evaluated model failed validation: {len(triggered)}",
    ]
    for winner in triggered[: config.max_groups_in_prompt]:
        reasons = (winner.get("failure_reasons") or [])[: config.max_rejected_models_in_prompt]
        lines.append(
            f"- {winner.get('forecast_group')}: fallback={winner.get('fallback_model')}, "
            f"trigger={winner.get('fallback_trigger')}, "
            f"original_candidates={winner.get('original_candidates')}"
        )
        if reasons:
            lines.append(
                f"    Why each candidate failed: {[(r.get('model_name'), r.get('reason')) for r in reasons]}"
            )
    return "\n".join(lines)


# Build the full substitution mapping every prompt template can draw from
def render_prompt_variables(result: PipelineResult, config: LLMConfig) -> dict[str, Any]:
    return {
        "context": render_context(result, config),
        "run_id": result.run_id,
    }
