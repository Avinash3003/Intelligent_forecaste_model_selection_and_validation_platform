"""Run-level MLflow governance view (Section 6.11 / reference "run traceability").

Deliberately *not* the Results payload. Results answers "what happened for this
one key" — the decision, its chart, its narrative. This answers "what happened
in this run overall": how many keys were processed, what was logged to the
tracking store, and the per-key outcome roll-up an auditor scans before opening
any single key. Sharing a service between the two would force one of them to
carry the other's shape, so they stay separate readers over the same result.
"""

from __future__ import annotations

from typing import Any

from app.orchestration.exceptions import RunNotReadyError
from app.orchestration.executor import PipelineExecutor, get_pipeline_executor
from app.orchestration.schemas import JobStatus, PipelineExecutionResult
from app.schemas.mlflow_view import (
    MLflowRunDetail,
    MLflowRunSummary,
    ParameterEntry,
    PerKeyOutcome,
)


class MLflowViewService:
    """Builds the MLflow Experiments page payload for one run."""

    def __init__(self, executor: PipelineExecutor | None = None) -> None:
        self._executor = executor or get_pipeline_executor()

    def get_run(self, run_id: str) -> MLflowRunDetail:
        result = self._executor.get_result(run_id)
        if result.job_status is not JobStatus.COMPLETED:
            raise RunNotReadyError(
                f"Run '{run_id}' has no tracking record yet ({result.job_status.value})."
            )

        tracking = result.mlflow_info or {}
        meta = result.run_metadata or {}
        training = (result.metrics or {}).get("training") or {}

        # winner_model is keyed by group; drift_results shares those keys.
        outcomes = [
            self._outcome(group_id, winner, (result.drift_results or {}).get(group_id) or {}, result)
            for group_id, winner in sorted((result.winner_model or {}).items())
        ]
        accuracies = [o.accuracy for o in outcomes if o.accuracy is not None]

        return MLflowRunDetail(
            run_id=run_id,
            mlflow_run_id=tracking.get("run_id"),
            experiment=tracking.get("experiment_name"),
            tracking_uri=tracking.get("tracking_uri"),
            status=tracking.get("status"),
            dataset=meta.get("dataset_path"),
            summary=MLflowRunSummary(
                keys_processed=len(outcomes),
                models_trained=int(training.get("trained") or 0),
                fallback_used=sum(1 for o in outcomes if o.fallback_used),
                models_registered=int(tracking.get("models_registered") or 0),
                parameters_logged=int(tracking.get("parameters_logged") or 0),
                metrics_logged=int(tracking.get("metrics_logged") or 0),
                # Platform accuracy is 100 − WMAPE (Section 10); averaged across
                # keys only where a WMAPE actually exists, so keys with no
                # backtest do not silently drag the mean toward zero.
                average_accuracy=round(sum(accuracies) / len(accuracies), 2) if accuracies else None,
            ),
            parameters=self._parameters(meta, outcomes),
            per_key=outcomes,
        )

    def _outcome(
        self,
        group_id: str,
        winner: dict[str, Any],
        drift: dict[str, Any],
        result: PipelineExecutionResult,
    ) -> PerKeyOutcome:
        model = winner.get("model_name")
        wmape = self._wmape(result, group_id, model)
        return PerKeyOutcome(
            group_id=group_id,
            model=model,
            status=winner.get("final_selection_status") or "Unknown",
            fallback_used=bool(winner.get("fallback_used")),
            original_backtest_rank=winner.get("original_backtesting_rank"),
            final_rank=winner.get("final_rank"),
            composite_score=winner.get("composite_ranking_score"),
            drift_algorithm=drift.get("algorithm"),
            drift_statistic=drift.get("statistic"),
            threshold_method=drift.get("threshold_method"),
            threshold_value=drift.get("threshold_value"),
            drift_result=(drift.get("result") or {}).get("outcome") if isinstance(drift.get("result"), dict) else drift.get("result"),
            # Section 10 defines platform accuracy as 100 − WMAPE.
            accuracy=round(max(0.0, 100.0 - wmape), 2) if isinstance(wmape, (int, float)) else None,
        )

    def _wmape(self, result: PipelineExecutionResult, group_id: str, model: str | None) -> float | None:
        """The winning model's own backtest WMAPE for this group."""
        if not model:
            return None
        for row in (((result.metrics or {}).get("backtesting") or {}).get("results") or []):
            if row.get("group_id") == group_id and row.get("model_name") == model:
                return ((row.get("backtest") or {}).get("overall") or {}).get("wmape")
        return None

    def _parameters(self, meta: dict[str, Any], outcomes: list[PerKeyOutcome]) -> list[ParameterEntry]:
        """The run configuration, as MLflow recorded it.

        Only the handful of parameters that identify *this* run are surfaced —
        the full set (200+, one block per key/model) is an artifact, not a
        table anyone reads on a page.
        """
        config = meta.get("configuration") or {}
        entries = [
            ("selected_models", ", ".join(meta.get("selected_models") or []) or "all registered"),
            ("date_column", config.get("date_column")),
            ("target_column", config.get("target_column")),
            ("key_columns", ", ".join(config.get("key_columns") or []) or "single series"),
            ("aggregation_method", config.get("aggregation_method")),
            ("frequency", meta.get("frequency")),
            ("mode", meta.get("mode")),
            ("groups_selected", str(sum(1 for o in outcomes if not o.fallback_used))),
            ("groups_fallback", str(sum(1 for o in outcomes if o.fallback_used))),
        ]
        return [ParameterEntry(name=name, value=str(value)) for name, value in entries if value]

    def is_ready(self, run_id: str) -> bool:
        try:
            return self._executor.get_result(run_id).job_status is JobStatus.COMPLETED
        except Exception:  # noqa: BLE001 - callers treat this as "not viewable"
            return False


_service: MLflowViewService | None = None


def get_mlflow_view_service() -> MLflowViewService:
    global _service
    if _service is None:
        _service = MLflowViewService()
    return _service
