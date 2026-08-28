from pydantic import BaseModel, Field

from app.schemas.compute import ComputeSelection
from app.schemas.metadata import MetadataMapping


class DeploymentRequest(BaseModel):
    file_id: str | None = None
    dataset_name: str | None = None
    metadata: MetadataMapping
    selected_models: list[str] = Field(default_factory=list)
    # Model used when every evaluated model fails validation. Must be one of
    # `selected_models`; None falls back to the engine's configured default.
    fallback_model: str | None = None
    # Months to forecast forward (Section 3: minimum 12-month horizon,
    # extended by this platform down to 6 and up to 60).
    horizon: int = Field(12, ge=6, le=60)
    # Derived feature columns (lag_*/rolling_mean_*/calendar) for the
    # tree-based models — Priority C. `None` (the field omitted) means
    # "every supported feature", reproducing pre-existing behavior exactly;
    # validated against the authoritative registry in `deployment_service`
    # before this ever reaches a Runner.
    derived_features: list[str] | None = None
    # Where the run executes. Required for Databricks execution; the local
    # runner ignores it.
    compute: ComputeSelection | None = None


class DeploymentResponse(BaseModel):
    run_id: str
    status: str
    message: str


class StageStatus(BaseModel):
    label: str
    status: str
    # ISO 8601 UTC, as recorded by the engine (forecast_engine/core/
    # pipeline_context.py's StageRecord) — the frontend converts to IST for
    # display, exactly like every other timestamp in the app. None while a
    # stage has not started yet, or has started but not finished.
    started_at: str | None = None
    completed_at: str | None = None


class ComputeStatus(BaseModel):
    """Where the run's compute is, before the forecast engine starts.

    Deliberately NOT an eighth pipeline phase. The seven display phases are
    a view over the engine's own stages; Databricks starting a cluster is
    infrastructure that happens before any of them, so it is reported
    alongside the trail rather than inside it.

    Every value is derived from the run's real Databricks lifecycle state —
    nothing here is timed, estimated or assumed.
    """

    # "starting" | "ready" | "failed"
    state: str
    label: str
    message: str
    detail: str | None = None


class DeploymentStatus(BaseModel):
    id: str
    dataset: str
    status: str
    start_time: str
    duration: str
    progress: int
    current_stage: str
    estimated_remaining: str
    stages: list[StageStatus]
    # Present only while the run is waiting on, or has just acquired, its
    # Databricks compute — None for local runs and once the engine reports
    # its first stage, at which point the phases speak for themselves.
    compute: ComputeStatus | None = None
    # Populated only for a failed run, so the UI can show why it failed
    # rather than a generic message.
    error: str | None = None
    # Display names only. Always server-derived from the authenticated
    # principal behind the request that started/cancelled the run — never
    # accepted as input from a client, so neither can be spoofed.
    started_by: str | None = None
    cancelled_by: str | None = None
