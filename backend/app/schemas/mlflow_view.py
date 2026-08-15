"""Schemas for the MLflow Experiments page (run-level governance view)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MLflowRunSummary(BaseModel):
    keys_processed: int = 0
    models_trained: int = 0
    fallback_used: int = 0
    models_registered: int = 0
    parameters_logged: int = 0
    metrics_logged: int = 0
    # 100 − WMAPE, averaged over keys that have a backtest (Section 10).
    average_accuracy: float | None = None


class ParameterEntry(BaseModel):
    name: str
    value: str


class PerKeyOutcome(BaseModel):
    group_id: str
    model: str | None = None
    status: str
    fallback_used: bool = False
    original_backtest_rank: int | None = None
    final_rank: int | None = None
    composite_score: float | None = None
    drift_algorithm: str | None = None
    drift_statistic: float | None = None
    threshold_method: str | None = None
    threshold_value: float | None = None
    drift_result: str | None = None
    accuracy: float | None = None


class TuningInfo(BaseModel):
    """Whether a search actually ran for this (key, model), and its
    outcome — from `HyperparameterTuner`, never fabricated when absent."""

    tuned: bool = False
    reason: str | None = None
    strategy: str | None = None
    cv_splits: int | None = None
    best_score_mae: float | None = None
    candidates_evaluated: int | None = None


class HyperparameterRecord(BaseModel):
    """One (key, model) pair's hyperparameters and its evaluation outcome, read
    from the pipeline's own reports and never invented for display."""

    model_config = ConfigDict(protected_namespaces=())

    group_id: str
    model_name: str
    is_winner: bool = False
    is_fallback: bool = False
    status: str  # Winner | Rejected | Fallback | Eliminated | Failed | Skipped
    wmape: float | None = None
    rmse: float | None = None
    mae: float | None = None
    rank: int | None = None
    hyperparameters: dict[str, object] = {}
    tuning: TuningInfo | None = None
    # True only for a fallback winner, whose model is fitted at selection
    # time (Section 6.9) and never passes through the tuned training path
    # at all — so it has no hyperparameter record to show, and reporting
    # one would be fabricated.
    hyperparameters_unavailable_reason: str | None = None


class MLflowRunDetail(BaseModel):
    run_id: str
    mlflow_run_id: str | None = None
    experiment: str | None = None
    tracking_uri: str | None = None
    status: str | None = None
    dataset: str | None = None
    summary: MLflowRunSummary
    parameters: list[ParameterEntry] = []
    per_key: list[PerKeyOutcome] = []
    hyperparameters: list[HyperparameterRecord] = []
