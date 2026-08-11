"""Reshapes `forecast_engine`'s run summary into the standardized
`PipelineExecutionResult` envelope (Section 6.14).

`PipelineContext.summary()` (produced unmodified by `forecast_engine`,
Section 6.13's `--summary-out`) is already a complete, plain-JSON record of
one run — this module does not recompute or reinterpret any forecasting
result, it only regroups already-serialized fields into the envelope
Section 6.14 asks for (Run Metadata, Forecast Results, Metrics,
Explainability, Drift Results, Winner Model, MLflow Information, Execution
Summary). Every Runner funnels its summary through this one function, so
Local and Databricks execution can never disagree about the result shape.
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
            # Written by Stage 5 (Persist Curated Dataset) — absent if curated
            # storage was disabled for the run, which the dataset preview
            # treats as "nothing to show" rather than falling back to the
            # much larger raw upload.
            "curated_dataset_uri": summary.get("curated_dataset_uri"),
            # Written by Persist Winning Models — where each group's winning
            # fitted model was durably stored, and whether it was.
            "model_storage": _model_storage(summary.get("model_storage_results") or []),
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
    """Summarize where this run's winning models were persisted.

    `models_saved` counts models actually written, not groups attempted —
    a group whose winner could not be serialized is reported by its own
    entry in `by_group` with the reason, never absorbed into the count.
    """
    persisted = [item for item in results if item.get("persisted")]

    return {
        "models_saved": len(persisted),
        "groups_total": len(results),
        # The run's model directory, read back from a path that was really
        # written rather than recomputed from configuration — so it cannot
        # report a location the files are not actually in. None when the
        # run persisted nothing.
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
            # Fallback audit trail, carried through so the dashboard can
            # explain *why* a fallback produced the forecast.
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
