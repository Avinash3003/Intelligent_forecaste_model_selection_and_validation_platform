"""Runs the forecast engine as a Databricks job.

Same PipelineRunner interface as LocalRunner, so switching is EXECUTION_MODE
and nothing else. It only submits the already-deployed job — no engine code
runs in this process:

    submit()      stage dataset + config to the UC Volume, start the job
    get_status()  Databricks run state -> JobStatus
    get_result()  summary.json -> PipelineExecutionResult

Two choices worth knowing:
  - Staging reuses the existing UC volume over ADLS, so no new storage or
    secret is introduced.
  - Every run parameter travels in one JSON config file. A wheel task's
    argument list is fixed in the bundle, so an unset parameter would arrive
    as an empty string ("--horizon ''" crashes; "--models ''" trains a model
    named ""). A config file also carries multi-valued columns.
"""

from __future__ import annotations

import io
import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.orchestration.exceptions import ExecutionError, RunNotReadyError, UnknownRunError
from app.orchestration.mlflow_history import MLflowHistoryStore
from app.orchestration.result_mapper import map_summary_to_result
from app.orchestration.runner_base import PipelineRunner
from app.orchestration.schemas import (
    CancellationOutcome,
    ExecutionBackend,
    JobStatus,
    PipelineExecutionRequest,
    PipelineExecutionResult,
    RunListing,
)
from app.utils.errors import safe_detail

logger = logging.getLogger(__name__)

# How long /deploy waits for a run's inputs to finish staging before it
# answers anyway.
#
# Staging uploads the dataset to the UC Volume byte-for-byte: a real 17.3 MB
# dataset measured 56.97s against this workspace, past the frontend's 30s
# request timeout on its own. The client aborted while the backend kept
# working and often did submit the run — the user saw "the request took too
# long" for a forecast that was actually starting.
#
# Short enough that the request always returns well inside that timeout,
# long enough that a small dataset (and every in-memory test workspace)
# finishes staging inline and the caller sees a fully submitted run.
_SUBMIT_STAGING_GRACE_SECONDS = 3.0

# Databricks' own run vocabulary -> this platform's. Kept as data rather
# than an if-chain so the mapping is auditable at a glance, and so a state
# Databricks adds later fails visibly (falls through to RUNNING) instead of
# being silently misreported as success.
_RESULT_STATE_MAP: dict[str, JobStatus] = {
    "SUCCESS": JobStatus.COMPLETED,
    "SUCCESS_WITH_FAILURES": JobStatus.FAILED,
    "FAILED": JobStatus.FAILED,
    "TIMEDOUT": JobStatus.FAILED,
    "CANCELED": JobStatus.CANCELLED,
    "UPSTREAM_FAILED": JobStatus.FAILED,
    "UPSTREAM_CANCELED": JobStatus.CANCELLED,
    "MAXIMUM_CONCURRENT_RUNS_REACHED": JobStatus.FAILED,
    "DISABLED": JobStatus.FAILED,
    "EXCLUDED": JobStatus.FAILED,
}

_LIFE_CYCLE_MAP: dict[str, JobStatus] = {
    "QUEUED": JobStatus.PENDING,
    "PENDING": JobStatus.PENDING,
    "WAITING_FOR_RETRY": JobStatus.PENDING,
    "BLOCKED": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "TERMINATING": JobStatus.RUNNING,
    "SKIPPED": JobStatus.CANCELLED,
    "INTERNAL_ERROR": JobStatus.FAILED,
}


# The SDK is imported lazily throughout this module so that importing it
# never requires Databricks credentials or the SDK to be installed.
def _databricks_service_modules() -> tuple[Any, Any]:
    from databricks.sdk.service import compute as compute_sdk
    from databricks.sdk.service import jobs as jobs_sdk

    return compute_sdk, jobs_sdk


# Compute selection is required: a run must never be redirected elsewhere.
def _require_compute(compute: Any) -> Any:
    if compute is None:
        raise ExecutionError("Select the compute this forecast should run on before deploying.")
    if compute.mode == "existing_compute":
        if not (compute.cluster_id or "").strip():
            raise ExecutionError("The selected existing compute is missing its cluster id.")
    elif compute.mode == "new_job_compute":
        if compute.job_compute is None:
            raise ExecutionError("The selected job compute configuration is incomplete.")
    else:
        raise ExecutionError(f"Unknown compute selection '{compute.mode}'.")
    return compute


# The seven Databricks task boundaries, in dependency order. Mirrors
# forecast_engine/run_pipeline.py's PHASE_ORDER exactly — kept in sync by
# hand, since the two packages are process-isolated and never import each
# other (same convention as this file's NAMING CONTRACT for stage names).
# Each task_key doubles as the --stage value its wheel task is invoked
# with, so the DAG node Databricks shows and the phase the engine actually
# runs can never drift apart.
# Tags a cluster this app created for one run — never a real user's
# all-purpose cluster. compute_service excludes it from Existing Compute.
RUN_CLUSTER_TAG = "forecastiq_run_id"

TASK_KEYS: tuple[str, ...] = (
    "load_prepare",
    "build_series",
    "train_models",
    "evaluate_models",
    "explain_models",
    "rank_select",
    "publish_results",
)


# One job cluster every task attaches to, rather than one cluster each.
_SHARED_JOB_CLUSTER_KEY = "forecastiq_pipeline"

# The four per-run paths, as `{{job.parameters.x}}` references. They live in
# the job definition once; `run_now` supplies the actual values per run, so
# the definition itself never has to be rewritten for a new dataset.
_JOB_PARAMETER_REFS = {
    name: f"{{{{job.parameters.{name}}}}}"
    for name in ("dataset", "config", "summary_out", "live_status_out")
}


# The engine CLI arguments one task passes to the wheel entry point.
def _engine_parameters(parameters: dict[str, str], stage: str) -> list[str]:
    return [
        "--dataset", parameters["dataset"],
        "--config", parameters["config"],
        "--summary-out", parameters["summary_out"],
        "--live-status-out", parameters["live_status_out"],
        "--parallel-keys",
        "--stage", stage,
    ]


# Matches the "-cpu-ml" or "-ml" infix an ML-runtime preset's version
# string carries, e.g. "15.4.x-cpu-ml-scala2.12" -> "15.4.x-scala2.12".
# Verified against this project's own live cluster: the same DBR line
# (15.4) is offered as both "15.4.x-cpu-ml-scala2.12" (an ML-runtime
# preset) and "15.4.x-scala2.12" (this project's real all-purpose cluster,
# confirmed running with use_ml_runtime=True set independently of the
# string) — so stripping the infix is a mechanical, verified transform, not
# a guess at Databricks' naming scheme.
_ML_RUNTIME_INFIX_RE = re.compile(r"-(?:cpu-)?ml(?=-|$)")


def _standard_runtime_version(runtime_key: str) -> str:
    """The Standard (non-ML) equivalent of an ML-runtime version string.

    A Docker Container Services image supplies its own Python/dependency
    stack in place of the ML runtime's — pairing the two is
    self-contradictory, so a cluster carrying `docker_image` must use the
    plain Standard runtime line underneath it instead.
    """
    return _ML_RUNTIME_INFIX_RE.sub("", runtime_key)


def map_run_state(life_cycle_state: str | None, result_state: str | None) -> JobStatus:
    """Map a Databricks run state to JobStatus.

    Terminal runs are judged by result_state, since TERMINATED alone only
    means "stopped". Unknown states report RUNNING — claiming success for a
    state we do not understand is the one wrong answer.
    """
    life_cycle = (life_cycle_state or "").upper()
    result = (result_state or "").upper()

    if life_cycle == "TERMINATED":
        return _RESULT_STATE_MAP.get(result, JobStatus.FAILED)
    return _LIFE_CYCLE_MAP.get(life_cycle, JobStatus.RUNNING)


@dataclass
class _DatabricksJobRecord:
    """One submission, holding only what Databricks and MLflow cannot supply.

    Does not survive a restart; finished runs come back from MLflow.
    """

    run_id: str
    status: JobStatus
    started_at: str
    dataset_name: str | None = None
    databricks_run_id: int | None = None
    databricks_run_url: str | None = None
    # Distinguishes "never tried" from "tried and Databricks had nothing
    # to give" — a plain `databricks_run_url is None` check cannot tell
    # those apart, which meant a run whose URL genuinely never resolves
    # (an SDK response with no run_page_url at all) triggered a fresh
    # jobs.get_run() call on every single poll forever, even long after
    # the run went terminal.
    databricks_run_url_attempted: bool = False
    dataset_uri: str | None = None
    summary_uri: str | None = None
    live_status_uri: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    started_by_user_id: str | None = None
    started_by_display_name: str | None = None
    cancelled_by_user_id: str | None = None
    cancelled_by_display_name: str | None = None
    # Decided once, when the run is submitted: which storage layout its
    # paths follow. Held here because cancellation has to delete exactly
    # what the run wrote, long after the request that chose it is gone.
    uses_container: bool = False


class DatabricksRunner(PipelineRunner):
    """Executes the pipeline as a run of the existing Databricks Job."""

    def __init__(
        self,
        settings: Settings,
        history: MLflowHistoryStore | None = None,
        workspace_client: Any | None = None,
        execution_backend: ExecutionBackend = ExecutionBackend.DATABRICKS,
    ) -> None:
        self._settings = settings
        self._jobs: dict[str, _DatabricksJobRecord] = {}
        self._lock = threading.Lock()
        # Identical history source to LocalRunner. Retargeting to a
        # Databricks-managed tracking server is `MLFLOW_TRACKING_URI`
        # changing, not a different reader — which is what makes run
        # history behave the same in both execution modes.
        self._history = history or MLflowHistoryStore(settings)
        self._client = workspace_client
        # Which reported backend and which deployed Job — everything else
        # (staging, submit/poll/retrieve, error translation) is identical
        # three-line subclass instead of a second implementation.
        self._execution_backend = execution_backend

    # ------------------------------------------------------------------
    # Workspace client
    # ------------------------------------------------------------------

    # Explicit, conservative SDK timeouts (Section 6.14 performance review).
    #
    # `_refresh()` calls `jobs.get_run()` on every status poll — from the
    # frontend's 3-second cadence, that is one call every few seconds for
    # as long as a run is active. The SDK's own defaults are tuned for a
    # one-off CLI invocation, not that cadence: `retry_timeout_seconds`
    # defaults to 300 (5 minutes), so one flaky call can hold a worker
    # thread retrying for far longer than the next poll is even five
    # minutes away — the poll loop was already going to try again in 3
    # seconds regardless. Trimming the retry budget to 30s does not make
    # retries more aggressive (backoff/attempt policy is unchanged, only
    # the ceiling is lower) — it just fails a stuck call fast enough that
    # "stuck" resolves on the *next* poll instead of on a five-minute wall
    # clock. `http_timeout_seconds` is left at the SDK's own default (60s)
    # explicitly rather than implicitly: `files.download()` of a large
    # run's `summary.json` is the one call on this client that can
    # legitimately take a while, and 60s per attempt is enough headroom for
    # that without this reasoning silently drifting if the SDK's default
    # ever changes.
    _WORKSPACE_HTTP_TIMEOUT_SECONDS = 60.0
    _WORKSPACE_RETRY_TIMEOUT_SECONDS = 30

    @property
    def _workspace(self) -> Any:
        """The SDK client, built lazily so importing this module never
        requires Databricks credentials to be configured."""
        if self._client is not None:
            return self._client

        host = (self._settings.databricks_host or "").strip()
        if not host:
            raise ExecutionError(
                "Databricks execution is selected but no workspace is configured. "
                "An administrator needs to set the Databricks connection settings."
            )

        try:
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.config import Config
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ExecutionError(
                "Databricks execution is unavailable because the Databricks SDK is not installed."
            ) from exc

        client_id = (self._settings.databricks_client_id or "").strip()
        client_secret = (self._settings.databricks_client_secret or "").strip()
        token = (self._settings.databricks_token or "").strip()

        try:
            if client_id and client_secret:
                # OAuth machine-to-machine with an Entra service principal:
                # workspace-scoped, rotatable, and not tied to any person.
                config = Config(
                    host=host,
                    client_id=client_id,
                    client_secret=client_secret,
                    http_timeout_seconds=self._WORKSPACE_HTTP_TIMEOUT_SECONDS,
                    retry_timeout_seconds=self._WORKSPACE_RETRY_TIMEOUT_SECONDS,
                )
            elif token:
                config = Config(
                    host=host,
                    token=token,
                    http_timeout_seconds=self._WORKSPACE_HTTP_TIMEOUT_SECONDS,
                    retry_timeout_seconds=self._WORKSPACE_RETRY_TIMEOUT_SECONDS,
                )
            else:
                raise ExecutionError(
                    "Databricks execution is selected but no workspace credential is configured. "
                    "An administrator needs to supply a service principal."
                )
            self._client = WorkspaceClient(config=config)
        except ExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
            raise ExecutionError(safe_detail(exc)) from exc

        return self._client

    # ------------------------------------------------------------------
    # PipelineRunner interface
    # ------------------------------------------------------------------

    def submit(self, request: PipelineExecutionRequest) -> str:
        run_id = request.run_id or f"dbx-run-{uuid.uuid4().hex[:12]}"
        record = _DatabricksJobRecord(
            run_id=run_id,
            status=JobStatus.PENDING,
            started_at=_now_iso(),
            dataset_name=request.dataset_name or Path(request.dataset_path).name,
            started_by_user_id=request.started_by_user_id,
            started_by_display_name=request.started_by_display_name,
            uses_container=self._uses_container_image(request.compute),
        )
        with self._lock:
            self._jobs[run_id] = record

        # Staging runs off the request thread, so /deploy answers in
        # milliseconds with the id the caller polls by.
        #
        # It cannot run inline: the dataset is uploaded to the UC Volume
        # byte-for-byte, and a real 17.3 MB dataset measured 56.97s against
        # this workspace — on its own already past the frontend's 30s
        # request timeout, before the config upload and jobs.submit that
        # follow it. The client aborted every time while the backend kept
        # going and often did submit the run, which is the worst of both:
        # the user sees "the request took too long" for a forecast that is
        # actually starting.
        #
        # Nothing about the run's semantics changes. The record is already
        # registered above, so a status poll finds it immediately; it simply
        # reports PENDING until staging finishes, which is exactly what
        # PENDING already means everywhere else in this runner.
        worker = threading.Thread(
            target=self._stage_and_trigger,
            args=(record, request),
            name=f"forecastiq-submit-{run_id}",
            daemon=True,
        )
        worker.start()
        # Bounded wait, not a fire-and-forget. Staging that finishes quickly
        # — a small dataset, or an in-memory workspace in tests — is fully
        # complete by the time this returns, so the run is already RUNNING
        # and its Databricks id already recorded. Staging that does not
        # finish in time keeps going on its own thread while the caller gets
        # its id now. Either way the caller polls the same record.
        worker.join(timeout=_SUBMIT_STAGING_GRACE_SECONDS)

        return run_id

    def _stage_and_trigger(
        self, record: _DatabricksJobRecord, request: PipelineExecutionRequest
    ) -> None:
        """Upload the run's inputs and start the Databricks run.

        Failures are recorded on the record rather than raised: by the time
        this runs the caller already holds the run id and is polling it, so
        a run that reports honestly why it never started beats one that
        vanishes.
        """
        run_id = record.run_id
        try:
            dataset_uri = self._upload_data_to_storage(run_id, request)
            databricks_run_id, _ = self._trigger_databricks_job(run_id, request, dataset_uri)
        except ExecutionError as exc:
            with self._lock:
                record.status = JobStatus.FAILED
                record.error = str(exc)
                record.completed_at = _now_iso()
            return
        except Exception as exc:  # noqa: BLE001 - a submit thread must never die silently
            logger.exception("Submitting run %s failed", run_id)
            with self._lock:
                record.status = JobStatus.FAILED
                record.error = safe_detail(exc, fallback="The forecast could not be started.")
                record.completed_at = _now_iso()
            return

        with self._lock:
            record.dataset_uri = dataset_uri
            run_artifacts_root = f"{self._artifacts_root(record.uses_container)}/{run_id}"
            record.summary_uri = f"{run_artifacts_root}/summary.json"
            record.live_status_uri = f"{run_artifacts_root}/live_status.json"
            record.databricks_run_id = databricks_run_id

        # A breadcrumb, not the source of truth: the in-memory record above
        # is that, but it does not survive a process restart, and MLflow
        # only has a run to restore once the engine's own tracking_pipeline
        # has started inside the job (see _DatabricksJobRecord's docstring).
        # A run still queued or booting compute at the moment of a restart
        # falls in the gap between those two — this is what closes it,
        # cheaply, once, right after Databricks has actually accepted the
        # submission.
        try:
            self._upload_run_file(
                f"{run_artifacts_root}/registry.json",
                json.dumps(
                    {
                        "run_id": run_id,
                        "databricks_run_id": databricks_run_id,
                        "started_at": record.started_at,
                        "dataset_name": record.dataset_name,
                        "started_by_user_id": record.started_by_user_id,
                        "started_by_display_name": record.started_by_display_name,
                    }
                ).encode("utf-8"),
            )
        except Exception:  # noqa: BLE001 - a missing breadcrumb only narrows recovery, never fails the run
            logger.warning("Could not write the run registry breadcrumb for %s", run_id)

    def get_status(self, run_id: str) -> JobStatus:
        record = self._find_job(run_id)
        if record is None:
            listing = self._history.get_listing(run_id)
            if listing is None:
                raise UnknownRunError(f"No run found for run_id '{run_id}'.")
            return listing.job_status

        self._refresh(record)
        return record.status

    def get_result(self, run_id: str) -> PipelineExecutionResult:
        record = self._find_job(run_id)
        if record is None:
            return self._restore_result(run_id)

        self._refresh(record)

        if record.status in (JobStatus.PENDING, JobStatus.RUNNING):
            raise RunNotReadyError(f"Run '{run_id}' is still {record.status.value}; poll get_status() first.")

        if record.status is JobStatus.COMPLETED and record.summary is None:
            record.summary = self._retrieve_pipeline_results(record)

        if record.summary is None:
            return PipelineExecutionResult(
                run_id=run_id,
                job_status=record.status,
                execution_backend=self._execution_backend,
                started_at=record.started_at,
                completed_at=record.completed_at,
                duration_seconds=record.duration_seconds,
                error=record.error,
            )

        return map_summary_to_result(
            run_id=run_id,
            execution_backend=self._execution_backend,
            job_status=record.status,
            summary=record.summary,
            started_at=record.started_at,
            completed_at=record.completed_at,
            duration_seconds=record.duration_seconds,
        )

    def _restore_result(self, run_id: str) -> PipelineExecutionResult:
        """Rebuild a finished run from MLflow, exactly as LocalRunner does.

        The job logs through the same engine code, so the summary is the
        same shape and uses the same mapper.
        """
        listing = self._history.get_listing(run_id)
        if listing is None:
            raise UnknownRunError(f"No run found for run_id '{run_id}'.")

        summary = self._history.get_summary(run_id) if listing.job_status is JobStatus.COMPLETED else None
        if summary is None:
            return PipelineExecutionResult(
                run_id=run_id,
                job_status=listing.job_status,
                execution_backend=self._execution_backend,
                started_at=listing.started_at,
                completed_at=listing.completed_at,
                duration_seconds=listing.duration_seconds,
                error=listing.error,
            )

        return map_summary_to_result(
            run_id=run_id,
            execution_backend=self._execution_backend,
            job_status=listing.job_status,
            summary=summary,
            started_at=listing.started_at,
            completed_at=listing.completed_at,
            duration_seconds=listing.duration_seconds,
        )

    def prewarm(self) -> None:
        self._history.prewarm()

    def list_runs(self) -> list[RunListing]:
        with self._lock:
            records = list(self._jobs.values())

        for record in records:
            self._refresh(record)

        active = [self._to_listing(record) for record in records]
        known = {listing.run_id for listing in active}
        merged = active + [listing for listing in self._history.list_runs() if listing.run_id not in known]
        merged.sort(key=lambda listing: listing.started_at or "", reverse=True)
        return merged

    def get_run(self, run_id: str) -> RunListing | None:
        record = self._find_job(run_id)
        if record is not None:
            self._refresh(record)
            return self._to_listing(record)

        listing = self._history.get_listing(run_id, with_stages=True)
        # One live call, bounded to a single-run detail read (never
        # list_runs): a listing rebuilt from MLflow history after a backend
        # restart knows its Databricks run id from the run's own tag but
        # has no resolved run_page_url yet.
        if listing is not None and listing.databricks_run_id is not None and listing.databricks_run_url is None:
            listing.databricks_run_url = self._run_page_url(listing.databricks_run_id)
        if listing is not None:
            return listing

        # Neither the in-memory record nor MLflow has this run: a restart
        # landed in the gap between "Databricks accepted the submission"
        # and "the engine's own tracking_pipeline logged the MLflow run" —
        # queued or still booting compute. The registry breadcrumb written
        # right after submission is the only thing left that knows this run
        # exists, so recover from it with one bounded live Jobs API call.
        return self._reconstruct_listing_from_registry(run_id)

    def _reconstruct_listing_from_registry(self, run_id: str) -> RunListing | None:
        registry = self._read_volume_json(f"{self._artifacts_root(False)}/{run_id}/registry.json")
        if registry is None:
            return None

        databricks_run_id = registry.get("databricks_run_id")
        if not isinstance(databricks_run_id, int):
            return None

        try:
            status, error, duration = self._monitor_job_status(databricks_run_id)
        except ExecutionError as exc:
            return RunListing(
                run_id=run_id,
                dataset_name=registry.get("dataset_name"),
                job_status=JobStatus.FAILED,
                execution_backend=self._execution_backend,
                started_at=registry.get("started_at") or _now_iso(),
                completed_at=None,
                duration_seconds=None,
                error=safe_detail(exc),
                databricks_run_id=databricks_run_id,
                databricks_run_url=self._run_page_url(databricks_run_id),
                started_by=registry.get("started_by_display_name"),
                cancelled_by=None,
                stages=[],
            )

        return RunListing(
            run_id=run_id,
            dataset_name=registry.get("dataset_name"),
            job_status=status,
            execution_backend=self._execution_backend,
            started_at=registry.get("started_at") or _now_iso(),
            # Unknown, not fabricated: _monitor_job_status reports duration
            # but not a completion timestamp, and a guessed "now" would be
            # wrong by however long this recovery path took to run.
            completed_at=None,
            duration_seconds=duration,
            error=error,
            databricks_run_id=databricks_run_id,
            databricks_run_url=self._run_page_url(databricks_run_id),
            started_by=registry.get("started_by_display_name"),
            cancelled_by=None,
            stages=[],
        )

    def cancel(
        self,
        run_id: str,
        cancelled_by_user_id: str | None = None,
        cancelled_by_display_name: str | None = None,
    ) -> CancellationOutcome:
        record = self._require_job(run_id)
        # Without this, a run that finished on Databricks since the last
        # poll still reads PENDING/RUNNING here (this record is only
        # updated on read, see _refresh's own docstring) — cancel() would
        # then proceed to delete a completed run's summary/models/forecast
        # via _cleanup_run_storage below and overwrite its real MLflow
        # outcome with CANCELLED.
        self._refresh(record)
        if record.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            # Already terminal — nothing to do, and nothing to report as a
            # failure. A second cancel() on the same run lands here.
            return CancellationOutcome(cancelled=False)

        cleanup_errors: list[str] = []

        if record.databricks_run_id is not None:
            try:
                self._workspace.jobs.cancel_run(run_id=record.databricks_run_id)
            except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
                # A request the control plane never accepted still leaves a
                # real job running — surfaced, not swallowed, but cleanup
                # below is still attempted for whatever the job had already
                # written before this call.
                cleanup_errors.append(f"Databricks job cancellation: {safe_detail(exc)}")

        cancelled_at = _now_iso()
        with self._lock:
            record.status = JobStatus.CANCELLED
            record.completed_at = cancelled_at
            record.cancelled_by_user_id = cancelled_by_user_id
            record.cancelled_by_display_name = cancelled_by_display_name

        # Requesting cancellation and the job actually stopping are not the
        # same instant — Databricks gives no synchronous "wait until fully
        # stopped" call, so there is an inherent, narrow race where the job
        # is still mid-write to a UC Volume file the instant cleanup below
        # runs. That is a property of asynchronous job cancellation, not
        # something this driver-side call can close without polling
        # `jobs.get_run()` in a loop — deliberately not done here, to avoid
        # turning a cancel request into a multi-second blocking call.
        cleanup_errors += self._cleanup_run_storage(run_id, record.uses_container)

        try:
            self._history.mark_cancelled(run_id, cancelled_by_user_id, cancelled_by_display_name, cancelled_at)
        except Exception as exc:  # noqa: BLE001 - report, never raise out of cancel()
            cleanup_errors.append(f"MLflow run lifecycle: {safe_detail(exc)}")

        return CancellationOutcome(cancelled=True, cleanup_errors=cleanup_errors)

    def _cleanup_run_storage(self, run_id: str, uses_container: bool) -> list[str]:
        """Delete every volume path this run wrote, scoped to runs/{run_id}.

        Idempotent — a missing path counts as already clean. Only the
        run-scoped copy is removed, never the original upload.
        Returns the locations that could not be removed; empty means all clean.
        """
        errors: list[str] = []
        for label, path in (
            ("uploaded dataset", self._run_root(run_id, uses_container)),
            ("curated dataset", f"{self._curated_root(run_id, uses_container)}/{run_id}"),
            ("trained models", f"{self._models_root(uses_container)}/{run_id}"),
            # Covers config, summary, live status and the registry
            # breadcrumb too — all under this same per-run folder now.
            ("run artifacts", f"{self._artifacts_root(uses_container)}/{run_id}"),
        ):
            try:
                self._delete_volume_directory(path)
            except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
                errors.append(f"{label} ({path}): {safe_detail(exc)}")

        forecast_path = f"{self._forecasts_root(uses_container)}/{run_id}_forecast.csv"
        try:
            self._workspace.files.delete(forecast_path)
        except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
            if not _is_not_found(exc):
                errors.append(f"forecast export ({forecast_path}): {safe_detail(exc)}")

        return errors

    def _delete_volume_directory(self, path: str) -> None:
        """Recursively empty and remove one volume directory.

        The Files API only deletes empty directories, so this walks
        depth-first. A missing directory counts as already clean, which is
        what makes a repeated cancel safe.
        """
        try:
            entries = list(self._workspace.files.list_directory_contents(path))
        except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
            if _is_not_found(exc):
                return
            raise

        for entry in entries:
            if entry.is_directory:
                self._delete_volume_directory(entry.path.rstrip("/"))
                self._workspace.files.delete_directory(entry.path.rstrip("/"))
            else:
                self._workspace.files.delete(entry.path)

        try:
            self._workspace.files.delete_directory(path)
        except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
            if not _is_not_found(exc):
                raise

    # ------------------------------------------------------------------
    # Integration points
    # ------------------------------------------------------------------

    def _upload_data_to_storage(self, run_id: str, request: PipelineExecutionRequest) -> str:
        """Stage the original dataset in the uploads volume, and the run's
        config alongside its other artifacts — never in uploads.

        Returns the staged dataset's volume path, which the job passes to --dataset.
        """
        source = Path(request.dataset_path)
        if not source.is_file():
            raise ExecutionError("The dataset for this run could not be found. Please upload it again.")

        uses_container = self._uses_container_image(request.compute)
        dataset_uri = f"{self._run_root(run_id, uses_container)}/{source.name}"
        config_uri = f"{self._artifacts_root(uses_container)}/{run_id}/forecast_configuration.json"

        try:
            self._upload_run_file(dataset_uri, source.read_bytes())
            self._upload_run_file(
                config_uri,
                json.dumps(self._job_configuration(run_id, request)).encode("utf-8"),
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
            raise ExecutionError(
                safe_detail(
                    exc,
                    fallback=(
                        "The dataset could not be staged for cloud execution. "
                        "An administrator needs to check the platform's storage access."
                    ),
                )
            ) from exc

        return dataset_uri

    def _job_configuration(self, run_id: str, request: PipelineExecutionRequest) -> dict[str, Any]:
        """The JSON the job hands the engine via --config: column mapping plus
        the per-run keys (run_id, models, fallback_model, horizon, dataset_name)."""
        payload: dict[str, Any] = dict(request.forecast_configuration)
        payload["run_id"] = run_id
        if request.dataset_name:
            payload["dataset_name"] = request.dataset_name
        if request.selected_models:
            payload["models"] = list(request.selected_models)
        if request.fallback_model:
            payload["fallback_model"] = request.fallback_model
        if request.derived_features is not None:
            payload["derived_features"] = list(request.derived_features)
        if request.horizon is not None:
            payload["horizon"] = int(request.horizon)
        if request.started_by_user_id:
            payload["started_by_user_id"] = request.started_by_user_id
        if request.started_by_display_name:
            payload["started_by_display_name"] = request.started_by_display_name
        # Curated output, resolved to an absolute UC Volume path here rather
        # than left to the engine's relative default — on a Databricks driver
        # that default resolves against a working directory the job destroys
        # on exit, so the curated dataset never outlived the run. Partitioned
        # by run id for the same reason the other outputs are: one run must
        # never overwrite another's.
        uses_container = self._uses_container_image(request.compute)
        payload["curated_storage"] = {"root_dir": self._curated_root(run_id, uses_container)}
        # Same reasoning for the winning models the run persists: a
        # relative root would put them on disposable driver storage.
        payload["model_storage"] = {"root_dir": self._models_root(uses_container)}
        # Same reasoning again for the exported forecast CSV and the
        # artifacts mirror — both outlive the run only if they land on a
        # UC Volume, not the driver's disposable working directory.
        payload["forecast_export"] = {"root_dir": self._forecasts_root(uses_container)}
        payload["artifacts_mirror"] = {"root_dir": self._artifacts_root(uses_container)}
        # No `volume_sync` block. Every path above is already the UC Volume
        # the data belongs in, so there is nothing left to copy — the engine
        # writes straight to the source of truth. `sync_outputs_to_volume`
        # returns None when the block is absent, so the old copy step is
        # inert without being deleted, and a single revert here restores it.
        return payload

    # The engine writes its outputs to four further roots. Under DCS these
    # cannot be UC Volumes either — the container has no `uc-volumes` scheme
    # handler at all, so every one of them fails the same way the run root
    # did, just later in the pipeline (Persist Curated rather than Load
    # Dataset). Each keeps its own sub-folder under the workspace staging
    # root so a DCS run's outputs stay as separated as the volumes keep
    # them, and every root still comes from settings.
    def _output_root(self, volumes_root: str, kind: str, uses_container: bool) -> str:
        # Same for both modes, for the same reason as _run_root: the
        # storage adapter decides how a volume is reached, so nothing here
        # needs to know which compute is running.
        return f"{volumes_root.rstrip('/')}/runs"

    def _forecasts_root(self, uses_container: bool) -> str:
        """Forecast output directory, without the run id — the writer adds it."""
        return self._output_root(self._settings.databricks_forecasts_volumes_root, "forecasts", uses_container)

    def _artifacts_root(self, uses_container: bool) -> str:
        """Artifact output directory, without the run id — the writer adds it."""
        return self._output_root(self._settings.databricks_artifacts_volumes_root, "artifacts", uses_container)

    def _curated_root(self, run_id: str, uses_container: bool) -> str:
        """Curated dataset directory, without the run id — the writer adds it."""
        return self._output_root(self._settings.databricks_curated_volumes_root, "curated", uses_container)

    def _models_root(self, uses_container: bool) -> str:
        """Winning-model directory, without the run id — the writer adds it."""
        return self._output_root(self._settings.databricks_models_volumes_root, "models", uses_container)

    def _trigger_databricks_job(
        self, run_id: str, request: PipelineExecutionRequest, dataset_uri: str
    ) -> tuple[int, str | None]:
        """Start one run of ForecastIQ's own Databricks Job.

        The job is a real, named, persistent definition this app owns and
        keeps current — seven tasks wired load_prepare -> build_series ->
        train_models -> evaluate_models -> explain_models -> rank_select ->
        publish_results by depends_on, sharing one JOB cluster. So every
        run lands in one job's run history, and its compute is billed at
        the Jobs rate and disappears when the run ends, instead of leaving
        an all-purpose cluster behind per run.

        Ray parallelism stays entirely inside train/evaluate/explain/
        rank_select's own tasks (see forecast_engine.parallel.ray_executor):
        the DAG is the orchestration graph, never one task per forecast key.
        """
        compute = _require_compute(request.compute)
        uses_container = self._uses_container_image(compute)
        run_artifacts_root = f"{self._artifacts_root(uses_container)}/{run_id}"
        parameters = {
            "dataset": dataset_uri,
            "config": f"{run_artifacts_root}/forecast_configuration.json",
            "summary_out": f"{run_artifacts_root}/summary.json",
            "live_status_out": f"{run_artifacts_root}/live_status.json",
        }

        compute_sdk, jobs_sdk = _databricks_service_modules()
        try:
            job_id = self._ensure_forecast_job(run_id, compute, compute_sdk, jobs_sdk)
            started = self._workspace.jobs.run_now(job_id=job_id, job_parameters=parameters)
            return int(started.run_id), None
        except ExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
            raise ExecutionError(
                safe_detail(exc, fallback="The forecast could not be started on the selected compute.")
            ) from exc

    # The job definition, created once and kept current afterwards.
    #
    # Reset before every run because compute is chosen per run and a job's
    # cluster spec is part of its definition — `run_now` cannot override
    # it. `max_concurrent_runs=1` is what makes that safe: a second run
    # queues behind the first rather than racing its reset.
    def _ensure_forecast_job(self, run_id: str, compute: Any, compute_sdk: Any, jobs_sdk: Any) -> int:
        settings = self._job_settings(run_id, compute, compute_sdk, jobs_sdk)
        job_id = self._find_forecast_job()
        if job_id is not None:
            self._workspace.jobs.reset(job_id=job_id, new_settings=jobs_sdk.JobSettings(**settings))
            return job_id

        # The ACL is set once, when the job is defined; every run of it is
        # then visible to the three role groups. A group that cannot be
        # resolved must not cost the run — create it unshared instead.
        acl = self._shared_run_acl(jobs_sdk)
        if acl:
            try:
                created = self._workspace.jobs.create(**settings, access_control_list=acl)
                return int(created.job_id)
            except Exception as exc:  # noqa: BLE001 - sharing must never block the run itself
                logger.warning(
                    "Could not share the ForecastIQ job with its role groups: %s", safe_detail(exc)
                )
        created = self._workspace.jobs.create(**settings)
        return int(created.job_id)

    # This app's own job, found by the name it created it under. Not the
    # old "resolve a pre-deployed job by name" routing: the definition
    # below is written by this runner and reset on every run, so the name
    # only identifies which job to keep updating.
    def _find_forecast_job(self) -> int | None:
        name = self._settings.databricks_job_display_name
        try:
            for job in self._workspace.jobs.list(name=name):
                return int(job.job_id)
        except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
            logger.warning("Could not look up the ForecastIQ job: %s", safe_detail(exc))
        return None

    # Seven tasks on one shared job cluster, with the per-run paths left as
    # job parameters `run_now` fills in.
    def _job_settings(self, run_id: str, compute: Any, compute_sdk: Any, jobs_sdk: Any) -> dict[str, Any]:
        libraries = self._engine_libraries(compute_sdk)
        tasks = []
        for index, task_key in enumerate(TASK_KEYS):
            task = jobs_sdk.Task(
                task_key=task_key,
                depends_on=(
                    [jobs_sdk.TaskDependency(task_key=TASK_KEYS[index - 1])] if index > 0 else None
                ),
                python_wheel_task=jobs_sdk.PythonWheelTask(
                    package_name="forecast_engine",
                    entry_point="forecast-engine",
                    parameters=_engine_parameters(_JOB_PARAMETER_REFS, task_key),
                ),
                libraries=libraries,
            )
            self._attach_compute(task, compute)
            tasks.append(task)

        settings: dict[str, Any] = {
            "name": self._settings.databricks_job_display_name,
            "tasks": tasks,
            "max_concurrent_runs": 1,
            "parameters": [
                jobs_sdk.JobParameterDefinition(name=name, default="")
                for name in _JOB_PARAMETER_REFS
            ],
        }
        if compute.mode != "existing_compute":
            settings["job_clusters"] = [
                jobs_sdk.JobCluster(
                    job_cluster_key=_SHARED_JOB_CLUSTER_KEY,
                    new_cluster=self._new_cluster_spec(run_id, compute.job_compute, compute_sdk),
                )
            ]
        return settings

    # Every run is a shared enterprise resource, not a personal one: whoever
    # submits it, the same three role groups can see it in Databricks
    # afterwards — the same visibility a persistent, permanently-shared job
    # would give, without giving up per-run compute selection (a real,
    # already-shipped feature a single static job's fixed task spec cannot
    # express — Databricks has no per-run-now cluster override).
    # `jobs.submit()` accepts an access_control_list on the one-time-run job
    # it creates, so this is one atomic call, not a submit-then-share race.
    # The selected compute is the compute that runs; there is no fallback.
    # new_job_compute attaches by job_cluster_key so all seven tasks share
    # the one job cluster defined alongside them; existing_compute attaches
    # the user's own cluster directly.
    def _attach_compute(self, task: Any, compute: Any) -> None:
        if compute.mode == "existing_compute":
            task.existing_cluster_id = compute.cluster_id
        else:
            task.job_cluster_key = _SHARED_JOB_CLUSTER_KEY

    def _shared_run_acl(self, jobs_sdk: Any) -> list[Any]:
        groups = [
            (self._settings.databricks_admins_group, jobs_sdk.JobPermissionLevel.CAN_MANAGE),
            (self._settings.databricks_datascientists_group, jobs_sdk.JobPermissionLevel.CAN_VIEW),
            (self._settings.databricks_analysts_group, jobs_sdk.JobPermissionLevel.CAN_VIEW),
        ]
        return [
            jobs_sdk.JobAccessControlRequest(group_name=group_name, permission_level=level)
            for group_name, level in groups
            if (group_name or "").strip()
        ]

    def _new_cluster_spec(self, run_id: str, config: Any, compute_sdk: Any) -> Any:
        cluster = compute_sdk.ClusterSpec(
            spark_version=config.runtime_key,
            node_type_id=config.node_type_id,
            data_security_mode=compute_sdk.DataSecurityMode.SINGLE_USER,
            custom_tags={RUN_CLUSTER_TAG: run_id},
            spark_env_vars=self._engine_cluster_env(),
        )
        self._attach_docker_image(cluster, compute_sdk)

        if config.autoscale:
            cluster.autoscale = compute_sdk.AutoScale(
                min_workers=config.min_workers, max_workers=config.max_workers
            )
            return cluster

        cluster.num_workers = config.num_workers
        if config.num_workers == 0:
            cluster.spark_conf = {
                "spark.databricks.cluster.profile": "singleNode",
                "spark.master": "local[*]",
            }
            cluster.custom_tags["ResourceClass"] = "SingleNode"
        return cluster

    # MLflow settings the engine needs ON the cluster, from this backend's
    # own configuration — never hardcoded here.
    #
    # A standard Databricks runtime pre-sets MLFLOW_TRACKING_URI itself, so
    # a run on one tracks to the workspace without help. A DCS image
    # replaces that environment, so without this the engine falls back to
    # MLflowConfig's local `sqlite:///mlflow.db` — tracking "succeeds" into
    # a file inside the container that dies with the cluster, and the run
    # never appears in the history this backend reads back.
    def _engine_cluster_env(self) -> dict[str, str]:
        forwarded = {
            "MLFLOW_TRACKING_URI": self._settings.mlflow_tracking_uri,
            "MLFLOW_REGISTRY_URI": self._settings.mlflow_registry_uri,
            "MLFLOW_EXPERIMENT_NAME": self._settings.mlflow_experiment_name,
        }
        return {key: value for key, value in forwarded.items() if (value or "").strip()}

    # Databricks Container Services: a new job cluster pulls the configured
    # image instead of resolving its dependencies from the runtime, when one
    # is configured. A blank URL is DCS staying off — the cluster is built
    # exactly as it always was, no docker_image on the spec at all. The URL
    # never comes from anywhere but settings, so there is nothing here for a
    # caller to override with its own value — the same guarantee
    # `_require_compute` already gives the cluster id and node type.
    #
    # Attaching the image alone is not sufficient. `spark_version` still
    # comes from an ML-runtime preset (compute_presets.RUNTIME_PRESETS), and
    # pairing an ML runtime with a Docker image is self-contradictory — the
    # image supplies the Python/dependency stack specifically INSTEAD OF the
    # ML runtime's own. So when DCS is on, this also downgrades the version
    # to its Standard (non-ML) equivalent.
    #
    # `use_ml_runtime` is deliberately left untouched, not forced to False —
    # verified against the real Jobs API (a live `jobs.submit` call), which
    # rejects the field outright when set explicitly on a spec with no
    # `kind`: "use_ml_runtime is not allowed with unspecified kind." Setting
    # `kind` (currently only `CLASSIC_PREVIEW`) is a bigger, preview-feature
    # decision this fix has no reason to force. Leaving the field unset and
    # downgrading only `spark_version` submits cleanly — Databricks infers
    # use_ml_runtime from the version string the same way it does for an ML
    # runtime, confirmed by that same live submission succeeding.
    def _attach_docker_image(self, cluster: Any, compute_sdk: Any) -> None:
        url = (self._settings.databricks_docker_image_url or "").strip()
        if not url:
            return

        username = (self._settings.databricks_docker_image_username or "").strip()
        password = (self._settings.databricks_docker_image_password or "").strip()
        basic_auth = (
            compute_sdk.DockerBasicAuth(username=username, password=password)
            if username and password
            else None
        )
        cluster.docker_image = compute_sdk.DockerImage(url=url, basic_auth=basic_auth)
        cluster.spark_version = _standard_runtime_version(cluster.spark_version)
        # A job-triggered SINGLE_USER cluster's run-as identity: this backend's
        # own service principal, the same one every other Databricks call in
        # this process authenticates as.
        owner = self._current_user_name()
        if owner:
            cluster.single_user_name = owner

    def _current_user_name(self) -> str | None:
        try:
            return self._workspace.current_user.me().user_name
        except Exception as exc:  # noqa: BLE001 - the cluster can still be attempted
            logger.warning("Could not resolve current user for cluster single_user_name: %s", exc)
            return None

    def _engine_libraries(self, compute_sdk: Any) -> list[Any]:
        wheel = (self._settings.databricks_engine_wheel_path or "").strip()
        return [compute_sdk.Library(whl=wheel)] if wheel else []

    def _run_page_url(self, databricks_run_id: int) -> str | None:
        """Databricks' own link to the run page, or None.

        Read from the API rather than assembled from a host and ids: the
        shape differs between a job run and a submitted run, and a guessed
        URL 404s for the user who trusted it.
        """
        try:
            return getattr(self._workspace.jobs.get_run(run_id=databricks_run_id), "run_page_url", None)
        except Exception:  # noqa: BLE001 - a missing link must never fail a poll
            return None

    def _monitor_job_status(self, databricks_run_id: int) -> tuple[JobStatus, str | None, float | None]:
        """Poll one run: returns its JobStatus, a safe error message, and duration."""
        try:
            run = self._workspace.jobs.get_run(run_id=databricks_run_id)
        except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
            raise ExecutionError(safe_detail(exc)) from exc

        state = getattr(run, "state", None)
        life_cycle = _enum_value(getattr(state, "life_cycle_state", None))
        result_state = _enum_value(getattr(state, "result_state", None))
        status = map_run_state(life_cycle, result_state)

        error: str | None = None
        if status is JobStatus.FAILED:
            # `state_message` is Databricks' own explanation and can carry
            # cluster URLs and secret-resolution internals, so it is
            # translated and redacted before it can reach a user.
            error = safe_detail(
                getattr(state, "state_message", "") or "",
                fallback="The forecast run failed in Azure Databricks.",
            )

        duration_ms = getattr(run, "run_duration", None) or getattr(run, "execution_duration", None)
        duration = float(duration_ms) / 1000.0 if duration_ms else None
        return status, error, duration

    def _retrieve_pipeline_results(self, record: _DatabricksJobRecord) -> dict[str, Any] | None:
        """Read the finished run's summary.json from the volume.

        Falls back to MLflow, which holds the identical summary, so a volume
        permission problem costs latency rather than the result.
        """
        payload = self._read_volume_json(record.summary_uri)
        if payload is not None:
            return payload
        return self._history.get_summary(record.run_id)

    def read_volume_text(self, uri: str) -> str | None:
        """A staged file as text, or None if unreadable.

        Public because a volume path means nothing on the API host, so the
        dataset preview must fetch it through this client.
        """
        payload = self._download_run_file(uri)
        return payload.decode("utf-8", errors="replace") if payload is not None else None

    def _download_run_file(self, uri: str) -> bytes | None:
        """Read one staged file back, through whichever API owns that path.

        A DCS run stages under /Workspace, everything else under a UC
        Volume; the writer already chose, so the reader must match.
        """
        try:
            if self._is_workspace_path(uri):
                with self._workspace.workspace.download(uri) as handle:
                    return handle.read()
            return self._workspace.files.download(uri).contents.read()
        except Exception:  # noqa: BLE001 - absence is normal, not an error
            return None

    def _read_volume_json(self, uri: str | None) -> dict[str, Any] | None:
        if not uri:
            return None
        # Not written yet (the job has not reached that stage), or not
        # readable. Both resolve on a later poll; neither is worth failing a
        # status request over.
        contents = self._download_run_file(uri)
        if contents is None:
            return None

        try:
            return json.loads(contents)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # Whether this run executes inside a Container Services image.
    #
    # DCS decides where a run can stage its files at all, so it is asked here
    # rather than inferred further down: a container cannot resolve /Volumes
    # (the runtime image that carries the `uc-volumes` scheme handler is the
    # very thing DCS replaces), while workspace files are reachable from
    # inside it.
    def _uses_container_image(self, compute: Any) -> bool:
        """Whether *this run* executes inside the custom container image.

        The image is attached to the cluster this runner creates for
        new_job_compute, and to nothing else — see _create_shared_cluster.
        An existing-compute run therefore executes on whatever runtime that cluster already has,
        with a working UC Volumes mount, no matter what
        DATABRICKS_DOCKER_IMAGE_URL is set to.

        Testing the setting alone (which is what this did) diverted those
        runs to workspace staging as well, quietly taking their outputs out
        of the storage account for no reason at all.
        """
        if getattr(compute, "mode", None) != "new_job_compute":
            return False
        return bool((self._settings.databricks_docker_image_url or "").strip())

    def _run_root(self, run_id: str, uses_container: bool) -> str:
        """One directory per run, holding ONLY its original uploaded
        dataset — config, summary, status and every other artifact live
        under `_artifacts_root` instead, never here.

        Always a UC Volume, for both execution modes. The container's lack
        of a `/Volumes` POSIX mount is handled where it belongs — in
        forecast_engine/core/storage.py, which reaches the same volume over
        the Files API — rather than by staging a second copy of the run's
        data somewhere the container can reach.

        `uses_container` is retained but no longer changes the answer: the
        old workspace-staging branch is kept one revert away while both
        execution modes are being proven end to end.
        """
        root = f"{self._settings.databricks_uploads_volumes_root.rstrip('/')}/runs"
        return f"{root.rstrip('/')}/{run_id}"

    # Workspace files and UC Volumes are different APIs on the same client.
    # One predicate decides which, so a path can never be written by one and
    # read back by the other.
    @staticmethod
    def _is_workspace_path(uri: str) -> bool:
        return uri.startswith("/Workspace/")

    def _upload_run_file(self, uri: str, payload: bytes) -> None:
        """Write one staged file, through whichever API owns that path."""
        if self._is_workspace_path(uri):
            from databricks.sdk.service.workspace import ImportFormat

            # The workspace API does not create parents the way the Files
            # API does — an upload into a folder that does not exist yet
            # fails with ResourceDoesNotExist rather than creating it. mkdirs
            # is idempotent, so this is safe on every run.
            parent = uri.rsplit("/", 1)[0]
            self._workspace.workspace.mkdirs(parent)
            # RAW, not AUTO: these are a dataset and a JSON config, not
            # notebooks. AUTO invites the workspace importer to interpret
            # them as source files.
            self._workspace.workspace.upload(
                path=uri, content=payload, format=ImportFormat.RAW, overwrite=True
            )
            return
        self._workspace.files.upload(uri, io.BytesIO(payload), overwrite=True)

    def _refresh(self, record: _DatabricksJobRecord) -> None:
        """Refresh one record from the workspace.

        Called on read rather than from a polling thread, so a backend with
        many historical runs does no background work.
        """
        if record.databricks_run_id is None:
            return

        # Resolved unconditionally, even for an already-terminal record.
        # This used to sit behind the same PENDING/RUNNING gate as the
        # status poll below, which meant a run that finished before the
        # frontend's first poll ever observed it in a non-terminal state
        # — e.g. Existing Compute against an already-warm cluster, fast
        # enough to complete inside one 3-second poll interval — got a
        # `databricks_run_url` that stayed None forever: nothing ever
        # fetched it, because the very next call hit the early return
        # below before reaching this line. The frontend then rendered no
        # "Open with Databricks" button at all for such a run — never a
        # broken link, just a silently missing one.
        #
        # Attempted exactly once: `databricks_run_url_attempted` is set
        # regardless of the outcome, so a genuinely absent URL (the SDK
        # response carried none) is remembered as "checked, nothing there"
        # rather than retried on every later poll — which is what a bare
        # `databricks_run_url is None` check did, and is exactly the
        # per-poll Jobs API call this method exists to avoid for a
        # terminal run.
        if not record.databricks_run_url_attempted:
            record.databricks_run_url = self._run_page_url(record.databricks_run_id)
            record.databricks_run_url_attempted = True

        if record.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            return

        try:
            status, error, duration = self._monitor_job_status(record.databricks_run_id)
        except ExecutionError as exc:
            # A transient workspace failure must not mark a healthy run as
            # failed — leave the record alone and let the next poll retry.
            record.error = str(exc)
            return

        # The engine writes its stage trail to the volume after every stage
        # transition, exactly as it does locally, so cloud runs show the
        # same live progress rather than a bare "Running".
        live = self._read_volume_json(record.live_status_uri)
        if live:
            record.stages = live.get("stages") or record.stages

        with self._lock:
            record.status = status
            if duration is not None:
                record.duration_seconds = duration
            if error:
                record.error = error
            if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED) and not record.completed_at:
                record.completed_at = _now_iso()

    def _to_listing(self, record: _DatabricksJobRecord) -> RunListing:
        stages = record.stages
        if record.summary is not None:
            stages = record.summary.get("stages") or stages
        return RunListing(
            run_id=record.run_id,
            dataset_name=record.dataset_name,
            job_status=record.status,
            execution_backend=self._execution_backend,
            started_at=record.started_at,
            completed_at=record.completed_at,
            duration_seconds=record.duration_seconds,
            error=record.error,
            databricks_run_url=record.databricks_run_url,
            started_by=record.started_by_display_name,
            cancelled_by=record.cancelled_by_display_name,
            stages=stages,
        )

    def _find_job(self, run_id: str) -> _DatabricksJobRecord | None:
        with self._lock:
            return self._jobs.get(run_id)

    def _require_job(self, run_id: str) -> _DatabricksJobRecord:
        record = self._find_job(run_id)
        if record is None:
            raise UnknownRunError(f"No run found for run_id '{run_id}'.")
        return record


def _is_not_found(exc: Exception) -> bool:
    """Whether `exc` means "that path does not exist" rather than a real
    failure — the expected, non-error outcome when cleanup reaches a
    location a run never wrote to, or a second cancel() finds already gone.
    """
    try:
        from databricks.sdk.errors import NotFound
    except ImportError:  # pragma: no cover - dependency is declared
        return False
    return isinstance(exc, NotFound)


def _enum_value(value: Any) -> str | None:
    """The string form of an SDK enum or a plain string — both must map alike."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
