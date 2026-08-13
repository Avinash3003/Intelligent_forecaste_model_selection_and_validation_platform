from datetime import datetime
from pathlib import Path

from app.auth.models import Principal
from app.orchestration.executor import PipelineExecutor, get_pipeline_executor
from app.orchestration.exceptions import UnknownRunError
from app.orchestration.schemas import JobStatus, PipelineExecutionRequest, RunListing
from app.schemas.deployment import DeploymentRequest, DeploymentResponse, DeploymentStatus, StageStatus
from app.services.upload_service import UploadService

# The forecast_engine's own stage names, in execution order. Used only to
# render the *shape* of the pipeline for a run that has not reported its
# stage trail yet (the engine emits that trail only in its final summary);
# a finished run always renders its real, recorded stages instead.
PIPELINE_STAGES = [
    "Load Dataset",
    "Detect Frequency",
    "Assess Data Quality",
    "Preprocess Dataset",
    "Persist Curated Dataset",
    "Verify Curated Dataset",
    "Generate Forecast Groups",
    "Build Forecast Series",
    "Train Models",
    "Evaluate Models",
    "Generate Explainability (SHAP)",
    "Rank & Select Production Models",
    "Generate Business Insights",
    "Track to MLflow",
]

_TERMINAL_STATUSES = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


def build_execution_request(
    request: DeploymentRequest, dataset_path: Path, principal: Principal
) -> PipelineExecutionRequest:
    """Reshape a `DeploymentRequest` (file_id + metadata mapping) into the
    orchestration layer's backend-and-transport-agnostic
    `PipelineExecutionRequest`.

    Shared between `DeploymentService` (the existing `/deploy` contract)
    and the `/execution/*` routes, so the two entry points never build this
    mapping differently.

    `principal` is the authenticated caller behind this HTTP request — the
    same object `Depends(require(...))` already produced for the route —
    never a value read from `request` itself. `DeploymentRequest` has no
    `started_by` field at all, so there is nothing here a client could ever
    override.
    """
    return PipelineExecutionRequest(
        dataset_path=str(dataset_path),
        dataset_name=request.dataset_name,
        forecast_configuration=request.metadata.model_dump(),
        selected_models=request.selected_models or None,
        fallback_model=request.fallback_model,
        horizon=request.horizon,
        derived_features=request.derived_features,
        started_by_user_id=principal.subject,
        started_by_display_name=principal.display_name or principal.subject,
        started_by_email=principal.email,
    )


class DeploymentService:
    """Submits a forecasting run through the Pipeline Executor
    (Section 6.14) — this is the *only* thing this service calls to make a
    run happen; it never imports training, evaluation, ranking, drift
    validation, MLflow, or LLM code directly.

    `list_deployments` reports the active Runner's real job registry. That
    registry is in-memory, so history covers the current backend process
    only and does not survive a restart — an honest limitation, and still
    strictly better than inventing runs that never happened.
    """

    def __init__(self, executor: PipelineExecutor | None = None, upload_service: UploadService | None = None) -> None:
        self._executor = executor or get_pipeline_executor()
        self._upload_service = upload_service or UploadService()

    def deploy(self, request: DeploymentRequest, principal: Principal) -> DeploymentResponse:
        """Submit one run and return the id it can be polled by.

        Raises:
            FileResolutionError: the `file_id` names no staged dataset.
            ExecutionError: the Runner refused the submission.

        Both propagate to the route rather than being answered with a
        synthetic run id: an id that refers to no real job would 404 on the
        very next status poll, which is worse than a clear failure here.
        """
        dataset_path, original_filename = self._upload_service.resolve(request.file_id)
        execution_request = build_execution_request(request, dataset_path, principal)
        execution_request.dataset_name = execution_request.dataset_name or original_filename

        run_id = self._executor.execute(execution_request)
        status = self._executor.get_status(run_id)

        return DeploymentResponse(
            run_id=run_id,
            status=status.value,
            message=f"Run submitted to the Pipeline Executor and is now {status.value}.",
        )

    def list_deployments(self) -> list[DeploymentStatus]:
        return [self._to_deployment_status(listing) for listing in self._executor.list_runs()]

    def get_deployment(self, run_id: str) -> DeploymentStatus:
        """One run's status detail.

        Raises:
            UnknownRunError: if no run with this id exists on the active
                Runner, so the route can answer 404 rather than inventing
                an empty record.
        """
        listing = self._executor.get_run(run_id)
        if listing is None:
            raise UnknownRunError(f"No run found for run_id '{run_id}'.")
        return self._to_deployment_status(listing)

    def _to_deployment_status(self, listing: RunListing) -> DeploymentStatus:
        stages = self._to_stage_statuses(listing)
        completed_stages = sum(1 for stage in stages if stage.status == "Completed")

        # A completed run is 100% by definition. A still-running run's live
        # stage trail only contains stages *begun so far* — it grows one
        # entry at a time as the engine reaches each one — so measuring
        # against its own length would read e.g. "9 of 10 stages done" (90%)
        # the instant stage 10 starts, when 4 of the real 14 stages haven't
        # even been reached yet. The true, fixed pipeline shape
        # (PIPELINE_STAGES) is the only correct denominator here.
        if listing.job_status is JobStatus.COMPLETED:
            progress = 100
        elif stages:
            progress = int(round(completed_stages / len(PIPELINE_STAGES) * 100))
        else:
            progress = 0

        return DeploymentStatus(
            id=listing.run_id,
            dataset=listing.dataset_name or "—",
            status=listing.job_status.value,
            # Raw ISO 8601 UTC — timezone/locale formatting (IST, 24-hour)
            # happens once, frontend-side, so every timestamp in the app is
            # guaranteed to use the same conversion rather than each screen
            # (or the backend and a screen) doing it slightly differently.
            start_time=listing.started_at,
            duration=_format_duration(listing, stages),
            progress=progress,
            current_stage=_current_stage(listing, stages),
            # No estimate is produced rather than a fabricated one: the
            # engine reports no per-stage timing until a run has finished.
            estimated_remaining="0 min" if listing.job_status is JobStatus.COMPLETED else "—",
            stages=stages,
            error=listing.error,
            started_by=listing.started_by,
            cancelled_by=listing.cancelled_by,
        )

    def _to_stage_statuses(self, listing: RunListing) -> list[StageStatus]:
        # A finished (or currently executing) run carries its real,
        # recorded stage trail — timestamps included, straight from the
        # engine's own StageRecord.
        if listing.stages:
            return [
                StageStatus(
                    label=stage.get("name", "—"),
                    status=stage.get("status", "Pending"),
                    started_at=stage.get("started_at"),
                    completed_at=stage.get("completed_at"),
                )
                for stage in listing.stages
            ]

        # Still executing (or a backend that reports no trail): show the
        # pipeline's shape with nothing yet claimed as done.
        if listing.job_status in _TERMINAL_STATUSES:
            return []
        return [StageStatus(label=label, status="Pending") for label in PIPELINE_STAGES]


# Human-readable run duration, measured live while a run is still going
def _format_duration(listing: RunListing, stages: list[StageStatus]) -> str:
    seconds = listing.duration_seconds
    if seconds is None and listing.job_status not in _TERMINAL_STATUSES:
        # No recorded duration yet — measure against the clock so a running
        # job shows elapsed time rather than a placeholder.
        try:
            started = datetime.fromisoformat(listing.started_at)
            seconds = (datetime.now(started.tzinfo) - started).total_seconds()
        except (TypeError, ValueError):
            seconds = None

    if seconds is None:
        return "—"
    # Never render a negative elapsed time: a started_at clock ahead of this
    # process's own would otherwise surface as "-278 min".
    seconds = max(seconds, 0)
    if seconds < 60:
        return f"{int(seconds)} sec"
    return f"{int(round(seconds / 60))} min"


# The stage a run is currently on, or the one it stopped at
def _current_stage(listing: RunListing, stages: list[StageStatus]) -> str:
    for stage in stages:
        if stage.status == "Running":
            return stage.label
    for stage in reversed(stages):
        if stage.status in ("Completed", "Failed"):
            return stage.label

    if listing.job_status is JobStatus.PENDING:
        return "Queued"
    if listing.job_status is JobStatus.RUNNING:
        return "In progress"
    return "—"
