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
from forecast_engine.s11_llm.security import sanitize_columns, sanitize_for_prompt


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
        f"Source data grain: {meta.get('frequency')}; mode: {meta.get('mode')}",
        # Without this the narrative described the source grain — "daily
        # sales" for a run whose every forecast and metric is monthly. The
        # grain is a platform invariant (Section 6.2: forecasting is month
        # level only, finer input is aggregated at ingestion), so it is
        # stated rather than inferred from the detected frequency.
        (
            "Forecasting grain: Monthly. Every forecast, horizon step and metric in this run is "
            "monthly; data finer than monthly was aggregated at ingestion. Describe the forecast "
            "as monthly regardless of the source grain above."
        ),
        # User-supplied column names (Section 13.1, "Guardrails for
        # injection/leakage") — the one place free text a user typed into
        # the wizard reaches an LLM prompt, so it is sanitized here rather
        # than trusted like every other value in this section.
        f"Date column: {sanitize_for_prompt(cfg.get('date_column'))}; "
        f"Target column: {sanitize_for_prompt(cfg.get('target_column'))}",
        f"Key columns: {sanitize_columns(cfg.get('key_columns'))}; "
        f"Feature columns: {sanitize_columns(cfg.get('feature_columns'))}",
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


# ----------------------------------------------------------------------
# Per-group context (Section 13.1 structured output; one call per group)
# ----------------------------------------------------------------------
#
# `render_context` above renders the *entire run* — every group's backtest,
# ranking and drift results in one block — because the legacy free-text
# engine wrote one narrative covering every group in a single call. The
# structured engine calls once per group instead (Section 6.1 Task 10:
# "Explainable AI Generation... Final model x key"), so it needs a context
# block scoped to exactly one group: small enough to keep token cost
# proportional to what is actually being explained, and unambiguous enough
# that the model has no other group's numbers to confuse this one's with.


def render_group_context(result: PipelineResult, config: LLMConfig, group_id: str) -> str:
    """The context block for one forecast group's structured insight call."""
    sections = [
        _dataset_section(result),
        _group_backtest_section(result, group_id),
        _group_forward_validation_section(result, group_id),
        _group_explainability_section(result, group_id, config),
        _group_ranking_section(result, group_id),
        _group_drift_section(result, group_id),
        _group_winner_section(result, group_id, config),
    ]
    return "\n\n".join(section for section in sections if section)


def _group_backtest_section(result: PipelineResult, group_id: str) -> str:
    items = [
        item for item in result.backtesting_metrics.get("results", []) if item.get("group_id") == group_id
    ]
    if not items:
        return ""
    lines = ["## Backtesting Metrics (Section 6.4)"]
    for item in items:
        overall = (item.get("backtest") or {}).get("overall") or {}
        lines.append(
            f"- {item.get('model_name')}: status={item.get('status')}, accuracy={overall.get('accuracy')}, "
            f"wmape={overall.get('wmape')}, rmse={overall.get('rmse')}, mae={overall.get('mae')}"
        )
    return "\n".join(lines)


def _group_forward_validation_section(result: PipelineResult, group_id: str) -> str:
    items = [
        item for item in result.forward_validation_results.get("results", []) if item.get("group_id") == group_id
    ]
    if not items:
        return ""
    lines = ["## Forward Forecast Validation (Section 6.5)"]
    for item in items:
        reasons = item.get("rejection_reasons") or []
        lines.append(
            f"- {item.get('model_name')}: {item.get('status')}"
            + (f" — rejected for: {', '.join(reasons)}" if reasons else "")
        )
    return "\n".join(lines)


def _group_explainability_section(result: PipelineResult, group_id: str, config: LLMConfig) -> str:
    items = [
        item for item in result.explainability_results.get("results", []) if item.get("group_id") == group_id
    ]
    if not items:
        return ""
    lines = ["## SHAP / Feature Importance (Section 6.10)"]
    for item in items:
        top_features = sorted(
            (item.get("global_importance") or {}).items(), key=lambda pair: pair[1], reverse=True
        )[: config.max_important_features]
        lines.append(f"- {item.get('model_name')}: method={item.get('method')}, top_features={top_features}")
    return "\n".join(lines)


def _group_ranking_section(result: PipelineResult, group_id: str) -> str:
    models = result.ranking_results.get("rankings", {}).get(group_id)
    if not models:
        return ""
    lines = ["## Model Ranking (Section 6.6)"]
    for model in models:
        lines.append(
            f"    {model.get('final_composite_rank')}. {model.get('model_name')} "
            f"(composite={model.get('composite_score')}, backtest_rank={model.get('original_backtest_rank')})"
        )
    return "\n".join(lines)


def _group_drift_section(result: PipelineResult, group_id: str) -> str:
    drift = result.drift_results.get(group_id)
    if not drift:
        return ""
    threshold = result.threshold_results.get(group_id, {})
    drift_result = drift.get("result") or {}
    return (
        "## Drift Detection & Threshold Estimation (Sections 6.7-6.9)\n"
        f"algorithm={drift.get('algorithm')}, statistic={drift.get('statistic')}, "
        f"threshold_method={threshold.get('method')}, threshold_value={threshold.get('value')}, "
        f"passed={drift_result.get('passed')}"
    )


def _group_winner_section(result: PipelineResult, group_id: str, config: LLMConfig) -> str:
    winner = next((w for w in result.final_winner_models if w.get("forecast_group") == group_id), None)
    if not winner:
        return ""
    rejected = (winner.get("rejected_candidates") or [])[: config.max_rejected_models_in_prompt]
    lines = [
        "## Final Production Model Selection (Section 6.9)",
        f"model={winner.get('final_production_model')}, status={winner.get('final_selection_status')}, "
        f"fallback={winner.get('fallback_flag')}, composite_score={winner.get('composite_ranking_score')}, "
        f"final_rank={winner.get('final_rank')}",
    ]
    if rejected:
        lines.append(f"Rejected candidates: {[(c.get('model_name'), c.get('reason')) for c in rejected]}")
    if winner.get("fallback_flag"):
        reasons = (winner.get("failure_reasons") or [])[: config.max_rejected_models_in_prompt]
        lines.append(f"Configured fallback model: {result.fallback_model}")
        lines.append(f"Fallback trigger: {winner.get('fallback_trigger')}")
        if reasons:
            lines.append(f"Why each original candidate failed: {[(r.get('model_name'), r.get('reason')) for r in reasons]}")
    return "\n".join(lines)


def group_decision_facts(result: PipelineResult, group_id: str) -> dict[str, Any]:
    """The plain facts one group's structured insight is built from —
    consumed by both the LLM prompt-variable builder and the deterministic
    template fallback, so the two paths are guaranteed to agree on what
    "the facts" are.
    """
    winner = next((w for w in result.final_winner_models if w.get("forecast_group") == group_id), {})
    model_name = winner.get("final_production_model") or "—"
    is_fallback = bool(winner.get("fallback_flag"))

    backtest_item = next(
        (
            item
            for item in result.backtesting_metrics.get("results", [])
            if item.get("group_id") == group_id and item.get("model_name") == model_name
        ),
        {},
    )
    overall = (backtest_item.get("backtest") or {}).get("overall") or {}
    wmape = _as_float(overall.get("wmape"))

    rejected = [
        {"model_name": c.get("model_name"), "reason": c.get("reason")}
        for c in (winner.get("rejected_candidates") or [])
        if c.get("model_name")
    ]
    if is_fallback:
        rejected = [
            {"model_name": r.get("model_name"), "reason": r.get("reason")}
            for r in (winner.get("failure_reasons") or [])
            if r.get("model_name")
        ]

    caveats: list[str] = []
    series = next((g for g in result.forecast_groups if g.get("group_id") == group_id), {})
    if not series.get("meets_minimum_history", True):
        caveats.append("short history")
    if is_fallback:
        caveats.append("fallback model used")

    # A simple, always-available confidence: backtest accuracy alone. The
    # richer 3-component score (backtest + forecast stability + drift
    # margin) lives in the backend (`app/services/confidence.py`) because
    # forecast stability needs the *rendered* forecast series, which this
    # engine-side facts extractor does not carry. The dashboard's displayed
    # confidence uses that richer backend score, not this field — this one
    # exists so the LLM/template has *something* to reference while writing
    # about confidence, never as the number of record.
    confidence_estimate = round(100.0 - wmape, 1) if wmape is not None else None

    return {
        "group_id": group_id,
        "selected_model": model_name,
        "wmape": wmape,
        "is_fallback": is_fallback,
        "fallback_trigger": winner.get("fallback_trigger"),
        "rejected_candidates": rejected,
        "confidence_estimate": confidence_estimate,
        "caveats": caveats,
    }


def group_grounding_metrics(result: PipelineResult, group_id: str) -> dict[str, float | int | None]:
    """Every number that would legitimately appear in this group's
    insight — the fact set `grounding.check_grounding` verifies claims
    against.
    """
    facts = group_decision_facts(result, group_id)
    metrics: dict[str, float | int | None] = {
        "wmape": facts["wmape"],
        "confidence_estimate": facts["confidence_estimate"],
    }

    backtest_item = next(
        (
            item
            for item in result.backtesting_metrics.get("results", [])
            if item.get("group_id") == group_id and item.get("model_name") == facts["selected_model"]
        ),
        {},
    )
    overall = (backtest_item.get("backtest") or {}).get("overall") or {}
    for key in ("mape", "rmse", "mae", "smape", "accuracy"):
        metrics[key] = _as_float(overall.get(key))

    ranking = result.ranking_results.get("rankings", {}).get(group_id) or []
    winning_rank = next((m for m in ranking if m.get("model_name") == facts["selected_model"]), {})
    metrics["composite_score"] = _as_float(winning_rank.get("composite_score"))
    metrics["final_rank"] = _as_float(winning_rank.get("final_composite_rank"))
    metrics["original_backtest_rank"] = _as_float(winning_rank.get("original_backtest_rank"))

    drift = result.drift_results.get(group_id) or {}
    metrics["drift_statistic"] = _as_float(drift.get("statistic"))
    threshold = result.threshold_results.get(group_id) or {}
    metrics["drift_threshold"] = _as_float(threshold.get("value"))

    return metrics


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
