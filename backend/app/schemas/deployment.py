from pydantic import BaseModel, Field
from app.config.run_limits import DEFAULT_FORECAST_HORIZON, MAX_FORECAST_HORIZON, MIN_FORECAST_HORIZON

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
    # Months to forecast forward. Bounds and default come from
    # app.config.run_limits, the one place this platform's horizon rule is
    # defined — see that module for why it is duplicated from the engine
    # rather than imported.
    horizon: int = Field(DEFAULT_FORECAST_HORIZON, ge=MIN_FORECAST_HORIZON, le=MAX_FORECAST_HORIZON)
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


class ParallelTask(BaseModel):
    """One key's real Ray task within a parallel stage."""

    group_id: str
    status: str
    # None only for a task that never got the chance to finish (Failed).
    duration_seconds: float | None = None
    worker_id: str | None = None
    node_id: str | None = None
    # Seconds from this stage's own start — a Gantt chart's x=0, not epoch
    # time. None for a Failed task, which has no real span to report.
    start: float | None = None
    end: float | None = None


class ParallelTaskSummary(BaseModel):
    """A stage's genuine parallel-execution shape, from the engine's own
    Ray fan-out — never inferred or estimated here."""

    executor: str
    total: int
    completed: int
    failed: int
    running: int
    max_concurrent: int | None = None
    tasks: list[ParallelTask] = Field(default_factory=list)


class StageStatus(BaseModel):
    label: str
    status: str
    # ISO 8601 UTC, as recorded by the engine (forecast_engine/core/
    # pipeline_context.py's StageRecord) — the frontend converts to IST for
    # display, exactly like every other timestamp in the app. None while a
    # stage has not started yet, or has started but not finished.
    started_at: str | None = None
    completed_at: str | None = None
    # The stage's own wall-clock boundary — from the engine's real Ray
    # fan-out (or the driver clock for a sequential stage), never another
    # stage's number reused. None only for a Pending phase.
    duration_seconds: float | None = None
    # One-line real outcome ("3 trained, 0 failed..."), from the engine's
    # own stage record — never fabricated here.
    detail: str | None = None
    # Set only for a stage that ran real Ray tasks across keys.
    parallel_tasks: ParallelTaskSummary | None = None


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
    stages: list[StageStatus]
    # Present only while the run is waiting on, or has just acquired, its
    # Databricks compute — None for local runs and once the engine reports
    # its first stage, at which point the phases speak for themselves.
    compute: ComputeStatus | None = None
    # Deep link to this run in the Databricks workspace, or None when one
    # is unavailable (a local run, or before Databricks has accepted the
    # submission). The UI renders the "Open with Databricks" action only
    # when this is set, so a run without a link still shows everything else.
    databricks_run_url: str | None = None
    # Populated only for a failed run, so the UI can show why it failed
    # rather than a generic message.
    error: str | None = None
    # Display names only. Always server-derived from the authenticated
    # principal behind the request that started/cancelled the run — never
    # accepted as input from a client, so neither can be spoofed.
    started_by: str | None = None
    cancelled_by: str | None = None
