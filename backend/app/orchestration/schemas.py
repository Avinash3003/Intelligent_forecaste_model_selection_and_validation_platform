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

    DATABRICKS submits to the Serverless job (the primary cloud path);
    DATABRICKS_DCS submits to the Container Services job (the ACR/Docker
    path, kept isolated — see `databricks_runner.py` and
    `docs/execution-modes.md`).
    """

    LOCAL = "local"
    DATABRICKS = "databricks"
    DATABRICKS_DCS = "databricks_dcs"


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
    # Derived feature columns for the tree-based models (Priority C) —
    # already validated against the authoritative registry by the caller
    # (deployment_service.build_execution_request) before this is built.
    derived_features: list[str] | None = None

    # Who submitted this run. Always derived server-side from the
    # authenticated `Principal` behind the `/deploy` or `/execution/submit`
    # request — never accepted from request-body JSON — so nothing here can
    # be spoofed by a caller naming a different user.
    started_by_user_id: str | None = None
    started_by_display_name: str | None = None
    started_by_email: str | None = None


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

    # Display names only — the stable user ids live on the Runner's own
    # job record / MLflow tags, not here. `RunListing` is what a run-history
    # view renders directly, and a view never needs more than a name.
    started_by: str | None = None
    cancelled_by: str | None = None

    stages: list[dict[str, Any]] = Field(default_factory=list)


class CancellationOutcome(BaseModel):
    """What actually happened when a cancellation was requested.

    `cancelled=False` means there was nothing to cancel (the run had
    already reached a terminal status) — not an error. `cleanup_errors` is
    deliberately a list, not a bool: a cancellation can succeed at stopping
    the run while individual storage locations fail to clean up, and a
    caller needs to know exactly which ones rather than a single opaque
    "cleanup failed".
    """

    cancelled: bool
    cleanup_errors: list[str] = Field(default_factory=list)


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
    # The detailed per-call LLM trace (Section 13.4) — one record per
    # attempt. `business_insights["trace_summary"]` carries only the
    # aggregate; this is the debuggable detail behind it, consumed by the
    # LLMOps observability view rather than the main Results dashboard.
    llm_trace: dict[str, Any] = Field(default_factory=dict)
    forecast_groups: list[dict[str, Any]] = Field(default_factory=list)
    execution_summary: dict[str, Any] = Field(default_factory=dict)

    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
