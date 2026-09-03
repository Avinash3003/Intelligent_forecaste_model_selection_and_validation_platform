from datetime import datetime
from pathlib import Path

from app.auth.models import Principal
from app.config.settings import Settings, get_settings
from app.config.model_availability import (
    compute_rejection_reason,
    container_runtime_required,
    unsupported_models,
)
from app.orchestration.executor import PipelineExecutor, get_pipeline_executor
from app.orchestration.exceptions import UnknownRunError
from app.orchestration.schemas import (
    ExecutionBackend,
    JobStatus,
    PipelineExecutionRequest,
    RunListing,
)
from app.schemas.deployment import (
    ComputeStatus,
    DeploymentRequest,
    DeploymentResponse,
    DeploymentStatus,
    ParallelTask,
    ParallelTaskSummary,
    StageStatus,
)

# The pipeline's stage vocabulary lives in pipeline_stages.py — shared with
# estimation, which needs the same names without importing the executor
# stack. Imported (not redefined) here because this module has always been
# where callers and tests reach for PIPELINE_STAGES.
from app.services.pipeline_stages import (
    PIPELINE_PHASES,
    PIPELINE_STAGES,
    STAGE_TO_PHASE,
    canonical_stage_name,
)
from app.services.upload_service import UploadService

_TERMINAL_STATUSES = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

_PIPELINE_STAGE_SET = frozenset(PIPELINE_STAGES)


class UnsupportedComputeError(ValueError):
    """The chosen compute is not a supported ForecastIQ execution runtime."""


class UnsupportedModelError(ValueError):
    """One or more requested models cannot run on the chosen compute."""


def _reject_unsupported_models(request: DeploymentRequest, settings: Settings) -> None:
    """Refuse a selection this compute cannot honour, naming each model.

    `uses_container` mirrors DatabricksRunner._uses_container_image: the
    image is attached to a job cluster this platform creates and to nothing
    else, so existing compute runs on whatever runtime that cluster has.
    """
    compute = request.compute
    reason = compute_rejection_reason(compute, settings)
    if reason:
        raise UnsupportedComputeError(reason)

    uses_container = (
        getattr(compute, "mode", None) == "new_job_compute"
        and bool((settings.databricks_docker_image_url or "").strip())
    )
    # With the container mandated, every accepted run carries the image, so
    # every model the image installs is available to every run.
    if container_runtime_required(settings):
        uses_container = True
    unsupported = unsupported_models(request.selected_models, uses_container)
    if unsupported:
        detail = "; ".join(f"{model}: {reason}" for model, reason in sorted(unsupported.items()))
        raise UnsupportedModelError(detail)

    _reject_unknown_runtime(compute)


def _reject_unknown_runtime(compute: object) -> None:
    """Refuse a runtime this platform does not offer, before a cluster boots.

    Databricks only rejects an unknown spark version when it starts the
    cluster, so a typo surfaced as "Invalid spark version" roughly five
    minutes into a run that was never going to work. The offered set is
    small and closed (the engine needs an ML runtime for Ray/xgboost/
    lightgbm/shap), so the answer is known at request time.

    Imported here rather than at module scope: compute_presets imports from
    app.schemas.compute, so a top-level import would close a cycle.
    """
    from app.config.compute_presets import RUNTIME_PRESETS

    config = getattr(compute, "job_compute", None)
    requested = (getattr(config, "runtime_key", "") or "").strip()
    if not requested:
        return

    offered = {runtime.key for runtime in RUNTIME_PRESETS}
    if requested not in offered:
        raise UnsupportedComputeError(
            f"Runtime {requested!r} is not one ForecastIQ offers. "
            f"Choose one of: {', '.join(sorted(offered))}."
        )


def build_execution_request(
    request: DeploymentRequest, dataset_path: Path, principal: Principal
) -> PipelineExecutionRequest:
    """Turn a DeploymentRequest into a transport-agnostic execution request.

    Shared by /deploy and /execution/* so the two never build it differently.
    principal is the authenticated caller, never read from the request body —
    DeploymentRequest has no started_by field for a client to override.
    """
    _reject_unsupported_models(request, get_settings())
    return PipelineExecutionRequest(
        dataset_path=str(dataset_path),
        dataset_name=request.dataset_name,
        forecast_configuration=request.metadata.model_dump(),
        # Passed through exactly as chosen. Models this compute cannot run
        # are refused before we get here (see _reject_unsupported_models),
        # never silently dropped: a run that quietly omits a model the user
        # picked is indistinguishable from one where the model lost.
        selected_models=list(request.selected_models) if request.selected_models else None,
        fallback_model=request.fallback_model,
        horizon=request.horizon,
        derived_features=request.derived_features,
        compute=request.compute,
        started_by_user_id=principal.subject,
        started_by_display_name=principal.display_name or principal.subject,
        started_by_email=principal.email,
    )


class DeploymentService:
    """Submits runs through the Pipeline Executor and reports their status.

    Never imports training, evaluation, ranking or MLflow code directly.
    list_deployments reads the Runner's in-memory registry, so it covers the
    current process only.
    """

    def __init__(self, executor: PipelineExecutor | None = None, upload_service: UploadService | None = None) -> None:
        self._executor = executor or get_pipeline_executor()
        self._upload_service = upload_service or UploadService()

    def deploy(self, request: DeploymentRequest, principal: Principal) -> DeploymentResponse:
        """Submit one run and return the id to poll it by.

        Raises FileResolutionError or ExecutionError rather than returning a
        made-up id, which would 404 on the very next poll.
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
        """One run's status. Raises UnknownRunError so the route can 404."""
        listing = self._executor.get_run(run_id)
        if listing is None:
            raise UnknownRunError(f"No run found for run_id '{run_id}'.")
        return self._to_deployment_status(listing)

    def _to_deployment_status(self, listing: RunListing) -> DeploymentStatus:
        stages = self._to_stage_statuses(listing)
        compute = _compute_status(listing, stages)

        # Counted against the ENGINE's stages, not the display phases:
        # the phases are a view for reading, and using them here would move
        # progress a whole phase at a time — and would sit wrongly against
        # the per-stage denominator below.
        completed_stages = sum(
            1
            for stage in (listing.stages or [])
            if canonical_stage_name(stage.get("name", "")) in _PIPELINE_STAGE_SET
            and stage.get("status") == "Completed"
        )

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
            # While compute is still being acquired there is no engine stage
            # to name, and "Queued" reads as if nothing is happening.
            current_stage=compute.label if compute else _current_stage(listing, stages),
            stages=stages,
            compute=compute,
            databricks_run_url=listing.databricks_run_url,
            error=listing.error,
            started_by=listing.started_by,
            cancelled_by=listing.cancelled_by,
        )

    def _to_stage_statuses(self, listing: RunListing) -> list[StageStatus]:
        """The stage trail, always covering the whole pipeline.

        The engine reports a stage only once it begins it, so a live run's
        trail grows one entry at a time. Its stages are rolled up into the
        display phases (pipeline_stages.PIPELINE_PHASES) — unreached phases
        stay visible as Pending rather than being dropped, and progress is
        still counted against the real stages so it does not jump a whole
        phase at a time.
        """
        reported = {
            canonical_stage_name(stage["name"]): stage
            for stage in (listing.stages or [])
            if stage.get("name")
        }

        # A terminal run with no trail at all has nothing to show — inventing
        # a column of Pending phases for a finished run would be a lie.
        if not reported and listing.job_status in _TERMINAL_STATUSES:
            return []

        # Rolled up to the display phases. A phase spans from the
        # first of its stages to start until the last to finish, so the
        # elapsed time it reports is real wall clock. Reporting each
        # stage's own engine time instead has understated a run's true
        # duration before — the gap is compute startup now (see
        # ComputeStatus), and was orchestration between DAG tasks on the
        # old Serverless job before that.
        phases: list[StageStatus] = []
        for label, stage_names in PIPELINE_PHASES:
            members = [reported[n] for n in stage_names if n in reported]

            if not members:
                phases.append(StageStatus(label=label, status="Pending"))
                continue

            starts = sorted(m["started_at"] for m in members if m.get("started_at"))
            ends = [m.get("completed_at") for m in members]
            statuses = [m.get("status", "Pending") for m in members]

            # A phase is only Completed once every stage in it is, and is
            # Failed the moment any is — a half-finished phase must never
            # render as done.
            if "Failed" in statuses:
                status = "Failed"
            elif len(members) == len(stage_names) and all(s == "Completed" for s in statuses):
                status = "Completed"
            else:
                status = "Running"

            phases.append(
                StageStatus(
                    label=label,
                    status=status,
                    started_at=starts[0] if starts else None,
                    completed_at=max(e for e in ends if e) if status == "Completed" and any(ends) else None,
                    duration_seconds=_phase_duration_seconds(members, starts, ends, status),
                    detail=_phase_detail(members),
                    parallel_tasks=_phase_parallel_tasks(members),
                )
            )

        # A stage the engine reported that no phase claims (a newly added or
        # renamed stage) is appended as its own row rather than dropped, so
        # the trail can never silently lose real, recorded execution — the
        # same guarantee the flat per-stage trail gave.
        phases.extend(
            StageStatus(
                label=name,
                status=stage.get("status", "Pending"),
                started_at=stage.get("started_at"),
                completed_at=stage.get("completed_at"),
            )
            for name, stage in reported.items()
            if name not in STAGE_TO_PHASE
        )

        return phases


# The phase's real work time, not driver wall clock where a better number
# exists. A Ray-parallel stage (Train/Evaluate/Explain/Rank & Select) has
# already finished all its real work by the time the driver opens it, so
# started_at..completed_at reads near-zero — measured_seconds carries the
# engine's own timing instead. Falls back to wall clock for phases with no
# measured value (sequential stages, where the driver's span is the work).
def _phase_duration_seconds(
    members: list[dict], starts: list[str], ends: list[str | None], status: str
) -> float | None:
    measured = [m["measured_seconds"] for m in members if m.get("measured_seconds") is not None]
    if measured:
        return round(sum(measured), 3)
    if status == "Completed" and starts and any(ends):
        from datetime import datetime

        start = datetime.fromisoformat(starts[0])
        end = datetime.fromisoformat(max(e for e in ends if e))
        return round((end - start).total_seconds(), 3)
    return None


# The phase's genuine Ray fan-out, straight from the engine's own
# per-stage telemetry — never inferred or estimated here. None for a
# phase that never ran independent parallel units.
def _phase_parallel_tasks(members: list[dict]) -> ParallelTaskSummary | None:
    for member in members:
        raw = member.get("parallel_tasks")
        if not raw:
            continue
        return ParallelTaskSummary(
            executor=raw.get("executor", "none"),
            total=raw.get("total_tasks", 0),
            completed=raw.get("completed_tasks", 0),
            failed=raw.get("failed_tasks", 0),
            running=raw.get("running_tasks", 0),
            max_concurrent=raw.get("max_concurrent_tasks"),
            tasks=[
                ParallelTask(
                    group_id=task["group_id"],
                    status=task.get("status", "Pending"),
                    duration_seconds=task.get("duration_seconds"),
                    worker_id=task.get("worker_id"),
                    node_id=task.get("node_id"),
                    start=task.get("start"),
                    end=task.get("end"),
                )
                for task in raw.get("tasks", [])
            ],
        )
    return None


# The phase's real outcome, from whichever of its stages last reported one —
# never fabricated here, only relayed.
def _phase_detail(members: list[dict]) -> str | None:
    for member in reversed(members):
        detail = member.get("detail")
        if detail:
            return detail
    return None


# Infrastructure progress, from the run's real Databricks lifecycle state.
#
# Databricks holds a submitted run in PENDING for exactly as long as it is
# acquiring compute — starting a stopped existing cluster, or provisioning a
# new job cluster. It reports RUNNING only once the task is actually
# executing on live compute. So the PENDING -> RUNNING transition IS the
# moment compute became ready: it is observed, never timed or guessed, and
# it is the same signal for both compute modes.
#
# Deliberately not a phase. The display phases are the engine's own
# stages, and this returns None the moment the engine reports the first of
# them, handing the trail back over.
def _compute_status(listing: RunListing, stages: list[StageStatus]) -> ComputeStatus | None:
    # Local execution has no compute to acquire.
    if listing.execution_backend is not ExecutionBackend.DATABRICKS:
        return None
    # The engine has started reporting: the phases below say it better.
    if any(stage.status != "Pending" for stage in stages):
        return None

    if listing.job_status is JobStatus.PENDING:
        return ComputeStatus(
            state="starting",
            label="Starting Compute",
            message="Starting the selected Databricks compute\u2026",
            detail="This may take a few minutes for a stopped cluster.",
        )
    if listing.job_status is JobStatus.RUNNING:
        return ComputeStatus(
            state="ready",
            label="Compute Ready",
            message="Databricks compute is ready. Starting the forecast pipeline\u2026",
        )
    # Failed before a single stage began — the compute itself never came up
    # (quota, a policy rejection, a cluster that could not start). Without
    # this the trail would show every phase Pending and no explanation.
    if listing.job_status is JobStatus.FAILED:
        return ComputeStatus(
            state="failed",
            label="Compute Unavailable",
            message="The Databricks compute for this run could not be started.",
            detail=listing.error,
        )
    return None


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
