"""The shared contracts every execution backend speaks.

One request shape in, one result shape out, whichever backend runs the
forecast — so routes and the frontend never branch on execution mode.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.compute import ComputeSelection


class JobStatus(str, Enum):
    """Run status, identical across backends so the UI renders one status pill."""

    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class ExecutionBackend(str, Enum):
    """Which backend runs the forecast — the EXECUTION_MODE setting."""

    LOCAL = "local"
    DATABRICKS = "databricks"


class PipelineExecutionRequest(BaseModel):
    """Everything a Runner needs to execute one run.

    Free of HTTP/upload concerns: the caller resolves file_id to a concrete
    dataset_path before building this.
    """

    run_id: str | None = None
    dataset_path: str
    # The name a user recognises; dataset_path points at the staged copy.
    dataset_name: str | None = None
    forecast_configuration: dict[str, Any]
    selected_models: list[str] | None = None
    fallback_model: str | None = None
    horizon: int | None = None
    # Lag/rolling/calendar features for the tree models, validated by the caller.
    derived_features: list[str] | None = None
    compute: ComputeSelection | None = None

    # Always taken from the authenticated user server-side, never from the
    # request body, so a caller cannot submit a run as someone else.
    started_by_user_id: str | None = None
    started_by_display_name: str | None = None
    started_by_email: str | None = None


class RunListing(BaseModel):
    """One run as the run-history view needs it.

    Thinner than PipelineExecutionResult so listing runs never deserializes
    every run's full forecast payload.
    """

    run_id: str
    dataset_name: str | None = None
    job_status: JobStatus
    execution_backend: ExecutionBackend

    started_at: str
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None

    # Databricks' own run page, as the API reports it. Not constructed
    # here: `run_page_url` is authoritative for both job and submitted
    # runs, and guessing a URL shape produces links that 404. None for
    # local runs and until Databricks has accepted the submission.
    databricks_run_url: str | None = None
    # The numeric id `databricks_run_url` above was resolved from, when a
    # caller has one but hasn't resolved the URL yet — a listing rebuilt
    # from MLflow history (backend restart) knows the id from the run's own
    # `databricks_run_id` tag but has no live client to turn it into a URL.
    # Resolving it is left to a caller that owns a workspace client, and
    # only on a single-run read: doing it per row in list_runs would be the
    # same N-live-calls-per-page mistake this codebase already paid for
    # once in the run-history sweep.
    databricks_run_id: int | None = None

    # Display names only — stable user ids live on the job record.
    started_by: str | None = None
    cancelled_by: str | None = None

    # The engine's stage records; grows as the run progresses.
    stages: list[dict[str, Any]] = Field(default_factory=list)


class CancellationOutcome(BaseModel):
    """Result of a cancel request.

    cancelled=False means there was nothing left to cancel, not an error.
    cleanup_errors names each storage location that could not be removed.
    """

    cancelled: bool
    cleanup_errors: list[str] = Field(default_factory=list)


class PipelineExecutionResult(BaseModel):
    """The finished run, in the one shape every backend returns.

    Nested payloads stay plain dicts — they are already the JSON the engine's
    own reports produce, so this is a reshaping of that summary, not a
    re-derivation of any forecast.
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
    # LLM narrative + per-group history, read by the Results dashboard.
    business_insights: dict[str, Any] = Field(default_factory=dict)
    # Per-call LLM detail behind business_insights["trace_summary"], read by
    # the Observability page.
    llm_trace: dict[str, Any] = Field(default_factory=dict)
    forecast_groups: list[dict[str, Any]] = Field(default_factory=list)
    execution_summary: dict[str, Any] = Field(default_factory=dict)

    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
