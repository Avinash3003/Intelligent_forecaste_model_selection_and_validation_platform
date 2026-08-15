"""Reshapes the engine's run summary into PipelineExecutionResult.

Pure regrouping of already-serialized fields — nothing here recomputes a
forecast. Every Runner funnels its summary through this one function so the
backends cannot disagree about the result shape.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from app.orchestration.schemas import ExecutionBackend, JobStatus, PipelineExecutionResult


def map_summary_to_result(
    run_id: str,
    execution_backend: ExecutionBackend,
    job_status: JobStatus,
    summary: dict[str, Any],
    started_at: str | None,
    completed_at: str | None,
    duration_seconds: float | None,
) -> PipelineExecutionResult:
    """Build the standardized result envelope from one run's summary JSON."""
    production_selection = summary.get("production_selection_report") or {}
    winners = production_selection.get("results", [])

    return PipelineExecutionResult(
        run_id=run_id,
        job_status=job_status,
        execution_backend=execution_backend,
        run_metadata={
            "run_id": summary.get("run_id"),
            "dataset_path": summary.get("dataset_path"),
            "configuration": summary.get("configuration"),
            "frequency": summary.get("frequency"),
            "mode": summary.get("mode"),
            "group_count": summary.get("group_count"),
            "series_count": summary.get("series_count"),
            "selected_models": summary.get("selected_models"),
            # Features the tree models were given. None = all supported
            # features, which is distinct from an explicit empty selection.
            "derived_features": summary.get("derived_features"),
            # From the dataset's own date column, not the run time.
            "date_range_start": (summary.get("quality_report") or {}).get("date_range_start"),
            "date_range_end": (summary.get("quality_report") or {}).get("date_range_end"),
            # Absent when curated storage was disabled for the run.
            "curated_dataset_uri": summary.get("curated_dataset_uri"),
            # Where each group's winning model was stored, and whether it was.
            "model_storage": _model_storage(summary.get("model_storage_results") or []),
            # The downloadable forecast CSV, or why it was not produced.
            "forecast_export": summary.get("forecast_export_result") or {},
            # Blob-accessible copy of the insights/LLM trace.
            "artifacts_mirror": summary.get("artifacts_mirror_result") or {},
        },
        forecast_results=summary.get("evaluation_report") or {},
        metrics={
            "training": summary.get("training_report"),
            "backtesting": summary.get("evaluation_report"),
            "ranking": summary.get("ranking_report"),
        },
        explainability=summary.get("explainability_report") or {},
        drift_results=_drift_results_by_group(winners),
        winner_model=_winner_model_by_group(winners),
        mlflow_info=summary.get("tracking_result") or {},
        # The generated narrative itself, not just its status — the Results
        # dashboard renders it directly.
        business_insights=summary.get("insight_report") or {},
        # Written by Generate Business Insights — one record per LLM call
        # attempt, consumed by the LLMOps observability view.
        llm_trace=summary.get("llm_trace") or {},
        # Per-group history (bounded tail) backing the actual-vs-forecast chart.
        forecast_groups=summary.get("forecast_groups") or [],
        execution_summary={
            "stages": summary.get("stages"),
            "metadata": summary.get("metadata"),
            "started_at": summary.get("started_at"),
            "completed_at": summary.get("completed_at"),
            "business_insights_status": (summary.get("insight_report") or {}).get("status"),
        },
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
    )


def _model_storage(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Where this run's winning models were persisted.

    models_saved counts models actually written, not groups attempted; a
    group that failed keeps its own entry in by_group with the reason.
    """
    persisted = [item for item in results if item.get("persisted")]

    return {
        "models_saved": len(persisted),
        "groups_total": len(results),
        # Read back from a path really written, so it cannot point at an
        # empty location. None when nothing was persisted.
        "location": str(PurePosixPath(persisted[0]["uri"]).parent) if persisted else None,
        "by_group": {
            item.get("forecast_group"): {
                "model_name": item.get("model_name"),
                "persisted": bool(item.get("persisted")),
                "uri": item.get("uri"),
                "error": item.get("error"),
            }
            for item in results
            if item.get("forecast_group")
        },
    }


def _drift_results_by_group(winners: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        winner["forecast_group"]: {
            "algorithm": winner.get("selected_drift_algorithm"),
            "statistic": winner.get("drift_statistic"),
            "threshold_method": winner.get("dynamic_threshold_method"),
            "threshold_value": winner.get("dynamic_threshold_value"),
            "result": winner.get("drift_validation_result"),
        }
        for winner in winners
    }


def _winner_model_by_group(winners: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        winner["forecast_group"]: {
            "model_name": winner.get("final_production_model"),
            "final_selection_status": winner.get("final_selection_status"),
            "fallback_used": winner.get("fallback_flag"),
            # Audit trail so the dashboard can explain why a fallback ran.
            "fallback_model": winner.get("fallback_model"),
            "fallback_trigger": winner.get("fallback_trigger"),
            "original_candidates": winner.get("original_candidates"),
            "failure_reasons": winner.get("failure_reasons"),
            "composite_ranking_score": winner.get("composite_ranking_score"),
            "final_rank": winner.get("final_rank"),
            "forecast": winner.get("forecast"),
        }
        for winner in winners
    }
