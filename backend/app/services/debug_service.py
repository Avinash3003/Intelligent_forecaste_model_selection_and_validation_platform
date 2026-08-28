"""A flat, traceable execution summary for developers.

Reads the same result the Results dashboard does, so the two can never
disagree — a different view of one source, not a second source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.orchestration.executor import PipelineExecutor, get_pipeline_executor
from app.orchestration.schemas import PipelineExecutionResult
from app.schemas.debug import ArtifactLocation, DebugSummary, GroupSelectionSummary, StatusCount

# Mirrors forecast_engine/s12_tracking/artifact_logger.py's `_PRODUCERS`
# registry verbatim — the fixed set of artifacts every successful run logs
# to MLflow, plus the consolidated run summary
# (forecast_engine/s12_tracking/run_summary.py). Kept here as a literal
# list rather than queried live: listing what an artifact store *contains*
# requires a real round-trip per run, while this fixed set is true for
# every run by construction (the same producer registry runs every time),
# and Section 6.11 asks for artifact *locations* as provenance, not a byte
# inventory.
KNOWN_ARTIFACT_PATHS = [
    ("Run summary (full result payload)", "run/summary.json"),
    ("Pipeline configuration", "configuration/pipeline_configuration.json"),
    ("Curated dataset", "dataset/curated_dataset_reference.json (or dataset/<file>)"),
    ("Forecast results", "forecast/forecast_results.json"),
    ("Metrics summary", "metrics/metrics_summary.json"),
    ("SHAP outputs", "explainability/shap_outputs.json"),
    ("Forecast plots", "plots/<group>_forecast.png"),
    ("Drift report", "drift/drift_report.json"),
    ("Ranking results", "ranking/ranking_results.json"),
    ("Fallback report", "selection/fallback_report.json"),
    ("LLM business summary", "insights/business_summary.json + .md"),
]


class DebugService:
    """Builds the developer debug summary for one run."""

    def __init__(self, executor: PipelineExecutor | None = None) -> None:
        self._executor = executor or get_pipeline_executor()

    def get_debug_summary(self, run_id: str) -> DebugSummary:
        """The debug summary.

        Unlike get_results this accepts an unfinished run — the whole point
        is seeing how far a RUNNING or FAILED execution got. Most fields are
        empty until the pipeline writes its reports; live stage progress
        belongs to GET /deployments/{run_id}, not here.
        """
        result = self._executor.get_result(run_id)
        meta = result.run_metadata or {}
        configuration = meta.get("configuration") or {}

        return DebugSummary(
            run_id=run_id,
            job_status=result.job_status.value,
            dataset_path=meta.get("dataset_path"),
            dataset_name=self._dataset_name(result),
            frequency=meta.get("frequency"),
            mode=meta.get("mode"),
            aggregation_method=configuration.get("aggregation_method"),
            forecast_group_count=meta.get("group_count"),
            series_count=meta.get("series_count"),
            selected_models=list(meta.get("selected_models") or []),
            fallback_model=self._fallback_model(result),
            training_summary=self._status_counts(
                (result.metrics or {}).get("training", {}).get("results", [])
            ),
            evaluation_summary=self._status_counts(
                (result.forecast_results or {}).get("results", [])
            ),
            winner_selection=self._winner_selection(result),
            mlflow_run_id=(result.mlflow_info or {}).get("run_id"),
            mlflow_experiment=(result.mlflow_info or {}).get("experiment_name")
            or (result.mlflow_info or {}).get("experiment_id"),
            mlflow_tracking_uri=(result.mlflow_info or {}).get("tracking_uri"),
            models_registered=(result.mlflow_info or {}).get("models_registered"),
            artifact_locations=[ArtifactLocation(name=name, path=path) for name, path in KNOWN_ARTIFACT_PATHS],
            execution_duration_seconds=result.duration_seconds,
            stages=(result.execution_summary or {}).get("stages") or [],
            key_execution=((result.execution_summary or {}).get("metadata") or {}).get("key_execution"),
            # mode="json" so nested enums (JobStatus, ExecutionBackend) are
            # already plain strings — this dict is embedded as-is in the
            # response, not re-validated through any model.
            raw_result=result.model_dump(mode="json"),
        )

    def _dataset_name(self, result: PipelineExecutionResult) -> str | None:
        # The display name a user typed at deploy time is recorded with the
        # run in MLflow (a tag) and in the Deployments list, but is not part
        # of the engine's own PipelineContext/summary output — so here, the
        # honest value is the dataset path's own filename, not a guess at a
        # field that does not exist on this payload.
        path = (result.run_metadata or {}).get("dataset_path")
        return Path(path).name if path else None

    def _fallback_model(self, result: PipelineExecutionResult) -> str | None:
        for winner in (result.winner_model or {}).values():
            model = winner.get("fallback_model")
            if model:
                return model
        return None

    def _status_counts(self, records: list[dict[str, Any]]) -> list[StatusCount]:
        counts: dict[str, int] = {}
        for record in records:
            status = record.get("status") or "Unknown"
            counts[status] = counts.get(status, 0) + 1
        return [StatusCount(status=status, count=count) for status, count in sorted(counts.items())]

    def _winner_selection(self, result: PipelineExecutionResult) -> list[GroupSelectionSummary]:
        return [
            GroupSelectionSummary(
                group_id=group_id,
                winner_model=winner.get("model_name"),
                selection_status=winner.get("final_selection_status") or "Unknown",
                fallback_used=bool(winner.get("fallback_used")),
            )
            for group_id, winner in (result.winner_model or {}).items()
        ]
