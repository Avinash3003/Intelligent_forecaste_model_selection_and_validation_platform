"""The run-level view: what happened across the whole run.

Deliberately separate from the Results payload, which answers "what happened
for this one key". This answers "how many keys, what was logged, how did each
come out" — the roll-up an auditor scans before opening any single key.
"""

from __future__ import annotations

from typing import Any

from app.orchestration.exceptions import RunNotReadyError
from app.orchestration.executor import PipelineExecutor, get_pipeline_executor
from app.orchestration.schemas import JobStatus, PipelineExecutionResult
from app.schemas.mlflow_view import (
    HyperparameterRecord,
    MLflowRunDetail,
    MLflowRunSummary,
    ParameterEntry,
    PerKeyOutcome,
    TuningInfo,
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
            hyperparameters=self._hyperparameter_records(result),
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

    def _hyperparameter_records(self, result: PipelineExecutionResult) -> list[HyperparameterRecord]:
        """Each (key, model) pair's final hyperparameters and its outcome.

        Read from reports the pipeline already produced. A model with no
        record (the fallback, fitted at selection time) is reported as
        unavailable rather than filled with a guess.
        """
        training_results = ((result.metrics or {}).get("training") or {}).get("results") or []
        backtest_by_key = {
            (row.get("group_id"), row.get("model_name")): row
            for row in ((result.metrics or {}).get("backtesting") or {}).get("results") or []
        }
        rankings = ((result.metrics or {}).get("ranking") or {}).get("rankings") or {}
        rank_by_key: dict[tuple[str, str], int] = {}
        for group_id, ranked in rankings.items():
            for entry in ranked:
                rank_by_key[(group_id, entry.get("model_name"))] = entry.get("final_composite_rank")
        winner_by_group = result.winner_model or {}

        records: list[HyperparameterRecord] = []
        seen_groups: set[str] = set()

        for row in training_results:
            group_id = row.get("group_id")
            model_name = row.get("model_name")
            training_status = row.get("status")
            winner = winner_by_group.get(group_id) or {}
            is_winner = training_status == "Trained" and winner.get("model_name") == model_name

            if is_winner:
                status = "Winner"
            elif training_status == "Trained":
                status = "Rejected"
            else:
                # Failed / Skipped / Unavailable — no hyperparameters exist
                # to show, but the attempt itself is still worth listing.
                status = training_status or "Unknown"

            backtest = backtest_by_key.get((group_id, model_name)) or {}
            overall = (backtest.get("backtest") or {}).get("overall") or {}
            tuning_raw = (row.get("metadata") or {}).get("tuning")

            records.append(
                HyperparameterRecord(
                    group_id=group_id,
                    model_name=model_name,
                    is_winner=is_winner,
                    is_fallback=False,
                    status=status,
                    wmape=overall.get("wmape"),
                    rmse=overall.get("rmse"),
                    mae=overall.get("mae"),
                    rank=rank_by_key.get((group_id, model_name)),
                    hyperparameters=row.get("params") or {} if training_status == "Trained" else {},
                    tuning=self._tuning_info(tuning_raw),
                )
            )
            if is_winner:
                seen_groups.add(group_id)

        # A fallback winner is fitted during Final Production Model
        # Selection (Section 6.9), never through the trained/tuned path
        # above, so it has no entry from the loop — added here, honestly
        # empty of hyperparameters rather than reusing another model's.
        for group_id, winner in winner_by_group.items():
            if group_id in seen_groups or not winner.get("fallback_used"):
                continue
            records.append(
                HyperparameterRecord(
                    group_id=group_id,
                    model_name=winner.get("model_name") or "unknown",
                    is_winner=True,
                    is_fallback=True,
                    status="Fallback",
                    hyperparameters={},
                    tuning=None,
                    hyperparameters_unavailable_reason=(
                        "The fallback model is fitted at selection time with fixed defaults, "
                        "not through the tuned training path — no hyperparameter record exists for it."
                    ),
                )
            )

        return records

    def _tuning_info(self, tuning_raw: Any) -> TuningInfo | None:
        if not isinstance(tuning_raw, dict):
            return None
        return TuningInfo(
            tuned=bool(tuning_raw.get("tuned")),
            reason=tuning_raw.get("reason"),
            strategy=tuning_raw.get("strategy"),
            cv_splits=tuning_raw.get("cv_splits"),
            best_score_mae=tuning_raw.get("best_score_mae"),
            candidates_evaluated=tuning_raw.get("candidates_evaluated"),
        )

    def _parameters(self, meta: dict[str, Any], outcomes: list[PerKeyOutcome]) -> list[ParameterEntry]:
        """The few parameters identifying this run; the full 200+ set is an
        artifact, not a table anyone reads on a page."""
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
