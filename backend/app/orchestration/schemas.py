"""Standardized orchestration contracts (Section 6.14).

`PipelineExecutionResult` is the *one* object every execution backend
returns. `LocalRunner` and `DatabricksRunner` never invent their own result
shape — the Pipeline Executor, and everything above it (FastAPI routes,
eventually the frontend), consumes exactly this, regardless of which
backend actually ran the forecast.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    """Internal execution status (Section 6.14, "Job Status").

    Local execution today only ever produces PENDING -> RUNNING ->
    (COMPLETED | FAILED | CANCELLED); a future Databricks Job reuses this
    exact vocabulary rather than inventing its own, which is what lets the
    frontend render one status pill regardless of backend.
    """

    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class ExecutionBackend(str, Enum):
    """Which Runner executed (or will execute) a run — Section 6.14's
    `execution.mode` configuration value.
    """

    LOCAL = "local"
    DATABRICKS = "databricks"


class PipelineExecutionRequest(BaseModel):
    """Everything a Runner needs to execute one forecasting run.

    Deliberately backend-and-frontend-agnostic: no `file_id`, no upload
    metadata — those are resolved to a concrete `dataset_path` and a plain
    `forecast_configuration` mapping *before* this object is built, by the
    caller (a backend service), so this package never needs to know about
    file staging or HTTP concerns.
    """

    run_id: str | None = None
    dataset_path: str
    # Human-readable name of the dataset being run, carried so a Runner can
    # report it back in `list_runs()` — `dataset_path` points at the staged
    # copy ("{file_id}_{name}.csv"), which is not what a user recognises.
    dataset_name: str | None = None
    forecast_configuration: dict[str, Any]
    selected_models: list[str] | None = None
    fallback_model: str | None = None
    horizon: int | None = None


class RunListing(BaseModel):
    """One submitted run as the Runner knows it — the raw material for a
    run-history view.

    Deliberately thinner than `PipelineExecutionResult`: listing every run
    must never require deserializing every run's full forecast payload.
    `stages` carries the engine's own stage records verbatim once a run has
    finished; it is empty while a run is still executing, because the
    engine reports its stage trail only in the final summary.
    """

    run_id: str
    dataset_name: str | None = None
    job_status: JobStatus
    execution_backend: ExecutionBackend

    started_at: str
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None

    stages: list[dict[str, Any]] = Field(default_factory=list)


class PipelineExecutionResult(BaseModel):
    """The one standardized result object every Runner returns
    (Section 6.14, "Pipeline Result").

    Every execution backend populates the same fields; a consumer (FastAPI
    route, frontend) never needs to know which backend actually produced
    this. Nested payloads are intentionally left as plain `dict` — they
    already carry the JSON-serializable shape `forecast_engine`'s own
    reports produce (`PipelineContext.summary()`), so this object is a
    reshaping of that summary into the envelope Section 6.14 asks for, not
    a re-derivation of any forecasting result.
    """

    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    job_status: JobStatus
    execution_backend: ExecutionBackend

    run_metadata: dict[str, Any] = Field(default_factory=dict)
    forecast_results: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    explainability: dict[str, Any] = Field(default_factory=dict)
    drift_results: dict[str, Any] = Field(default_factory=dict)
    winner_model: dict[str, Any] = Field(default_factory=dict)
    mlflow_info: dict[str, Any] = Field(default_factory=dict)
    # LLM narrative and per-group history, both consumed directly by the
    # Results dashboard.
    business_insights: dict[str, Any] = Field(default_factory=dict)
    forecast_groups: list[dict[str, Any]] = Field(default_factory=list)
    execution_summary: dict[str, Any] = Field(default_factory=dict)

    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
