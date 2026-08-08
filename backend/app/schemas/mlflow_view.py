"""Schemas for the MLflow Experiments page (run-level governance view)."""

from __future__ import annotations

from pydantic import BaseModel


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
