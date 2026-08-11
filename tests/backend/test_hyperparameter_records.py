"""Hyperparameters surfaced through the existing MLflow Experiments
endpoint — one record per (key, model), joined from the training,
backtesting, and ranking reports the pipeline already produces.

No new storage, no new endpoint: everything here is read from
`PipelineExecutionResult.metrics` / `.winner_model`, the same fields the
"Child runs by key" table already reads.
"""

from __future__ import annotations

from app.orchestration.exceptions import UnknownRunError
from app.orchestration.schemas import ExecutionBackend, JobStatus, PipelineExecutionResult
from app.services.mlflow_view_service import MLflowViewService


class _FakeExecutor:
    def __init__(self, results: dict[str, PipelineExecutionResult]) -> None:
        self._results = results

    def get_result(self, run_id: str) -> PipelineExecutionResult:
        if run_id not in self._results:
            raise UnknownRunError(f"No run found for run_id '{run_id}'.")
        return self._results[run_id]


def _training_row(group_id, model_name, status="Trained", params=None, tuning=None):
    return {
        "group_id": group_id,
        "model_name": model_name,
        "status": status,
        "params": params or {},
        "metadata": {"tuning": tuning} if tuning is not None else {},
    }


def _backtest_row(group_id, model_name, wmape=None, rmse=None, mae=None):
    return {
        "group_id": group_id,
        "model_name": model_name,
        "backtest": {"overall": {"wmape": wmape, "rmse": rmse, "mae": mae}},
    }


def _result(training_rows, backtest_rows=None, rankings=None, winner_model=None) -> PipelineExecutionResult:
    return PipelineExecutionResult(
        run_id="run-1",
        job_status=JobStatus.COMPLETED,
        execution_backend=ExecutionBackend.LOCAL,
        metrics={
            "training": {"results": training_rows},
            "backtesting": {"results": backtest_rows or []},
            "ranking": {"rankings": rankings or {}},
        },
        winner_model=winner_model or {},
    )


def _service(result: PipelineExecutionResult) -> MLflowViewService:
    return MLflowViewService(_FakeExecutor({"run-1": result}))


# ---------------------------------------------------------------------
# The winner gets the real hyperparameters that actually produced its score
# ---------------------------------------------------------------------


def test_a_tuned_winner_shows_its_real_best_params_and_matching_metrics():
    tuning = {
        "tuned": True,
        "strategy": "random",
        "cv_splits": 3,
        "best_params": {"n_estimators": 500},  # not part of TuningInfo, must not crash
        "best_score_mae": 12.3,
        "candidates_evaluated": 8,
    }
    result = _result(
        training_rows=[
            _training_row(
                "1 | 1", "xgboost", params={"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05}, tuning=tuning
            )
        ],
        backtest_rows=[_backtest_row("1 | 1", "xgboost", wmape=25.98, rmse=244.4, mae=194.4)],
        rankings={"1 | 1": [{"model_name": "xgboost", "final_composite_rank": 1}]},
        winner_model={"1 | 1": {"model_name": "xgboost", "fallback_used": False}},
    )

    detail = _service(result).get_run("run-1")

    assert len(detail.hyperparameters) == 1
    record = detail.hyperparameters[0]
    assert record.group_id == "1 | 1"
    assert record.model_name == "xgboost"
    assert record.is_winner is True
    assert record.status == "Winner"
    assert record.hyperparameters == {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05}
    assert record.wmape == 25.98
    assert record.rmse == 244.4
    assert record.mae == 194.4
    assert record.rank == 1
    assert record.tuning.tuned is True
    assert record.tuning.best_score_mae == 12.3
    assert record.tuning.candidates_evaluated == 8


def test_a_rejected_candidate_is_still_shown_with_its_own_params():
    result = _result(
        training_rows=[
            _training_row("1 | 1", "xgboost", params={"n_estimators": 500}),
            _training_row("1 | 1", "prophet", params={"seasonality_mode": "additive"}),
        ],
        backtest_rows=[_backtest_row("1 | 1", "prophet", wmape=31.42)],
        winner_model={"1 | 1": {"model_name": "xgboost", "fallback_used": False}},
    )

    detail = _service(result).get_run("run-1")

    prophet = next(r for r in detail.hyperparameters if r.model_name == "prophet")
    assert prophet.status == "Rejected"
    assert prophet.is_winner is False
    assert prophet.hyperparameters == {"seasonality_mode": "additive"}
    assert prophet.wmape == 31.42


def test_a_model_that_was_never_tuned_reports_tuned_false_with_a_reason():
    result = _result(
        training_rows=[
            _training_row(
                "1 | 1", "prophet", params={"seasonality_mode": "additive"},
                tuning={"tuned": False, "reason": "Model selects its parameters internally."},
            )
        ],
        winner_model={"1 | 1": {"model_name": "prophet", "fallback_used": False}},
    )

    record = _service(result).get_run("run-1").hyperparameters[0]

    assert record.tuning.tuned is False
    assert record.tuning.reason == "Model selects its parameters internally."


# ---------------------------------------------------------------------
# Fallback: no fabricated hyperparameters
# ---------------------------------------------------------------------


def test_a_fallback_winner_has_no_hyperparameters_and_says_why():
    """The fallback is fitted at selection time, never through training —
    there is no tuned/default param record for it anywhere, and none may
    be invented for display."""
    result = _result(
        training_rows=[_training_row("1 | 4", "xgboost", status="Failed", params={})],
        winner_model={"1 | 4": {"model_name": "seasonal_naive", "fallback_used": True}},
    )

    detail = _service(result).get_run("run-1")

    fallback = next(r for r in detail.hyperparameters if r.model_name == "seasonal_naive")
    assert fallback.status == "Fallback"
    assert fallback.is_fallback is True
    assert fallback.hyperparameters == {}
    assert fallback.hyperparameters_unavailable_reason is not None
    assert "fixed defaults" in fallback.hyperparameters_unavailable_reason


def test_a_failed_training_attempt_has_no_hyperparameters_shown():
    """A failed fit's `params` field is whatever was last set on the model
    object, not a real trained result — it must never be shown as if it
    were the parameters that produced a score."""
    result = _result(training_rows=[_training_row("1 | 1", "xgboost", status="Failed", params={"n_estimators": 300})])

    record = _service(result).get_run("run-1").hyperparameters[0]

    assert record.status == "Failed"
    assert record.hyperparameters == {}


# ---------------------------------------------------------------------
# Multiple keys / models stay correctly attributed
# ---------------------------------------------------------------------


def test_every_key_model_combination_is_attributed_correctly_not_mixed_up():
    result = _result(
        training_rows=[
            _training_row("1 | 1", "xgboost", params={"n_estimators": 500}),
            _training_row("1 | 2", "xgboost", params={"n_estimators": 200}),
        ],
        backtest_rows=[
            _backtest_row("1 | 1", "xgboost", wmape=10.0),
            _backtest_row("1 | 2", "xgboost", wmape=20.0),
        ],
        winner_model={
            "1 | 1": {"model_name": "xgboost", "fallback_used": False},
            "1 | 2": {"model_name": "xgboost", "fallback_used": False},
        },
    )

    detail = _service(result).get_run("run-1")

    by_key = {r.group_id: r for r in detail.hyperparameters}
    assert by_key["1 | 1"].hyperparameters == {"n_estimators": 500}
    assert by_key["1 | 1"].wmape == 10.0
    assert by_key["1 | 2"].hyperparameters == {"n_estimators": 200}
    assert by_key["1 | 2"].wmape == 20.0


def test_no_training_results_produces_an_empty_list_not_an_error():
    detail = _service(_result(training_rows=[])).get_run("run-1")

    assert detail.hyperparameters == []
