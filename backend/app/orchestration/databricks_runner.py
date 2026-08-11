"""Databricks Runner (Section 6.14) — real execution against Azure Databricks.

Implements the same `PipelineRunner` interface `LocalRunner` does, so
switching between them is `EXECUTION_MODE` changing and nothing else. What
it does is submit the *existing* bundle-deployed Job — it never reimplements
any part of the forecasting pipeline, and it never runs engine code in this
process:

    submit()
        -> _upload_data_to_storage()    dataset + config -> UC Volume (ADLS)
        -> _trigger_databricks_job()    volume paths -> Databricks run id
    get_status()
        -> _monitor_job_status()        Databricks run state -> JobStatus
    get_result()
        -> _retrieve_pipeline_results() summary.json -> PipelineExecutionResult

Two deliberate choices worth stating:

* **Staging goes through the existing Unity Catalog external volume**, not a
  new storage account or a second container. The volume is already backed by
  the deployment's ADLS `uploads` container, the job already reads POSIX
  paths from it, and the workspace credential the API already holds is
  enough — so no additional secret, and no duplicated storage, is introduced.

* **Every run parameter travels in one JSON config file**, not as individual
  job parameters. A `python_wheel_task`'s argument list is static in the
  bundle, so an unset parameter would reach the engine's argparse as an
  empty string — `--horizon ""` is a hard crash, `--models ""` silently
  trains a model named "". A config file has no such failure mode and is
  also the only way to express multi-valued key/feature columns.
"""

from __future__ import annotations

import io
import json
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
    ExecutionBackend,
    JobStatus,
    PipelineExecutionRequest,
    PipelineExecutionResult,
    RunListing,
)
from app.utils.errors import safe_detail

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


def map_run_state(life_cycle_state: str | None, result_state: str | None) -> JobStatus:
    """Translate a Databricks run state pair into this platform's `JobStatus`.

    A terminal run is judged by `result_state` — `TERMINATED` alone says
    only that the run stopped, not whether it succeeded. Anything
    unrecognised is reported as RUNNING rather than COMPLETED: claiming
    success for a state we do not understand is the one wrong answer.
    """
    life_cycle = (life_cycle_state or "").upper()
    result = (result_state or "").upper()

    if life_cycle == "TERMINATED":
        return _RESULT_STATE_MAP.get(result, JobStatus.FAILED)
    return _LIFE_CYCLE_MAP.get(life_cycle, JobStatus.RUNNING)


@dataclass
class _DatabricksJobRecord:
    """In-memory record of one submission.

    Holds only what cannot be re-read from Databricks or MLflow. Like
    `LocalRunner`'s registry this does not survive a backend restart;
    finished runs are recoverable from MLflow, which is what
    `_restore_result` uses.
    """

    run_id: str
    status: JobStatus
    started_at: str
    dataset_name: str | None = None
    databricks_run_id: int | None = None
    dataset_uri: str | None = None
    summary_uri: str | None = None
    live_status_uri: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)


class DatabricksRunner(PipelineRunner):
    """Executes the pipeline as a run of the existing Databricks Job."""

    def __init__(
        self,
        settings: Settings,
        history: MLflowHistoryStore | None = None,
        workspace_client: Any | None = None,
        execution_backend: ExecutionBackend = ExecutionBackend.DATABRICKS,
        job_name: str | None = None,
        job_id: int | None = None,
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
        # between Serverless and DCS, which is what lets `DcsRunner` be a
        # three-line subclass instead of a second implementation.
        self._execution_backend = execution_backend
        self._job_name = job_name or settings.databricks_job_name
        self._job_id: int | None = job_id if job_id is not None else settings.databricks_job_id

    # ------------------------------------------------------------------
    # Workspace client
    # ------------------------------------------------------------------

    @property
    def _workspace(self) -> Any:
        """The Databricks SDK client, built on first use.

        Constructed lazily so importing this module — which happens on
        every backend start, whatever the execution mode — never requires
        Databricks credentials to be present.
        """
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
                self._client = WorkspaceClient(host=host, client_id=client_id, client_secret=client_secret)
            elif token:
                self._client = WorkspaceClient(host=host, token=token)
            else:
                raise ExecutionError(
                    "Databricks execution is selected but no workspace credential is configured. "
                    "An administrator needs to supply a service principal."
                )
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
        )
        with self._lock:
            self._jobs[run_id] = record

        try:
            record.dataset_uri = self._upload_data_to_storage(run_id, request)
            record.summary_uri = f"{self._run_root(run_id)}/summary.json"
            record.live_status_uri = f"{self._run_root(run_id)}/live_status.json"
            record.databricks_run_id = self._trigger_databricks_job(run_id, request, record.dataset_uri)
        except ExecutionError as exc:
            # The submission is recorded as FAILED rather than discarded:
            # the frontend already holds this run id, and a run that 404s
            # on its very next status poll is a worse experience than one
            # that reports honestly why it never started.
            with self._lock:
                record.status = JobStatus.FAILED
                record.error = str(exc)
                record.completed_at = _now_iso()
            return run_id

        return run_id

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

        The Databricks job logs through the same engine code, so its
        consolidated summary artifact is byte-for-byte the same shape a
        local run produces and goes through the identical mapper.
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
        return self._history.get_listing(run_id, with_stages=True)

    def cancel(self, run_id: str) -> bool:
        record = self._require_job(run_id)
        if record.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            return False

        if record.databricks_run_id is not None:
            try:
                self._workspace.jobs.cancel_run(run_id=record.databricks_run_id)
            except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
                raise ExecutionError(safe_detail(exc)) from exc

        with self._lock:
            record.status = JobStatus.CANCELLED
            record.completed_at = _now_iso()
        return True

    # ------------------------------------------------------------------
    # Integration points
    # ------------------------------------------------------------------

    def _upload_data_to_storage(self, run_id: str, request: PipelineExecutionRequest) -> str:
        """Stage the dataset and its run configuration in the UC Volume.

        Both land under one per-run directory, so a run's inputs and
        outputs are one path prefix — which is what makes retention or
        cleanup a single delete rather than a hunt across locations.

        Returns:
            The POSIX volume path of the staged dataset, which is what the
            Databricks job passes to `--dataset`.
        """
        source = Path(request.dataset_path)
        if not source.is_file():
            raise ExecutionError("The dataset for this run could not be found. Please upload it again.")

        run_root = self._run_root(run_id)
        dataset_uri = f"{run_root}/{source.name}"
        config_uri = f"{run_root}/forecast_configuration.json"

        try:
            with source.open("rb") as handle:
                self._workspace.files.upload(dataset_uri, handle, overwrite=True)

            self._workspace.files.upload(
                config_uri,
                io.BytesIO(json.dumps(self._job_configuration(run_id, request)).encode("utf-8")),
                overwrite=True,
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
        """The single JSON payload the job hands the engine via `--config`.

        Carries the column mapping the engine has always accepted, plus the
        per-run keys it now also reads from this file (`run_id`, `models`,
        `fallback_model`, `horizon`, `dataset_name`). See this module's
        docstring for why these travel here rather than as job parameters.
        """
        payload: dict[str, Any] = dict(request.forecast_configuration)
        payload["run_id"] = run_id
        if request.dataset_name:
            payload["dataset_name"] = request.dataset_name
        if request.selected_models:
            payload["models"] = list(request.selected_models)
        if request.fallback_model:
            payload["fallback_model"] = request.fallback_model
        if request.horizon is not None:
            payload["horizon"] = int(request.horizon)
        # Curated output, resolved to an absolute UC Volume path here rather
        # than left to the engine's relative default — on a Databricks driver
        # that default resolves against a working directory the job destroys
        # on exit, so the curated dataset never outlived the run. Partitioned
        # by run id for the same reason the other outputs are: one run must
        # never overwrite another's.
        payload["curated_storage"] = {"root_dir": self._curated_root(run_id)}
        # Same reasoning for the winning models the run persists: a
        # relative root would put them on disposable driver storage.
        payload["model_storage"] = {"root_dir": self._models_root()}
        return payload

    def _curated_root(self, run_id: str) -> str:
        """The curated volume's run directory — *without* the run id.

        The engine's curated writer appends the run id itself, so adding it
        here too would nest it twice.
        """
        return f"{self._settings.databricks_curated_volumes_root.rstrip('/')}/runs"

    def _models_root(self) -> str:
        """The models volume's run directory — *without* the run id.

        Like the curated writer, the model writer appends the run id itself,
        which is also what keeps one run from overwriting another's models.
        """
        return f"{self._settings.databricks_models_volumes_root.rstrip('/')}/runs"

    def _trigger_databricks_job(
        self, run_id: str, request: PipelineExecutionRequest, dataset_uri: str
    ) -> int:
        """Start the existing forecast job and return its Databricks run id."""
        run_root = self._run_root(run_id)
        parameters = {
            "dataset": dataset_uri,
            "config": f"{run_root}/forecast_configuration.json",
            "summary_out": f"{run_root}/summary.json",
            "live_status_out": f"{run_root}/live_status.json",
        }

        try:
            waiter = self._workspace.jobs.run_now(job_id=self._resolve_job_id(), job_parameters=parameters)
            return int(waiter.run_id)
        except ExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
            raise ExecutionError(
                safe_detail(exc, fallback="The forecast could not be started in Azure Databricks.")
            ) from exc

    def _resolve_job_id(self) -> int:
        """The job id to run, by configured id or by name.

        Resolving by name is the default because redeploying the asset
        bundle mints a new job id — pinning one in configuration would
        break the API on every deployment. Cached after the first lookup.
        """
        if self._job_id is not None:
            return self._job_id

        name = self._job_name
        try:
            matches = list(self._workspace.jobs.list(name=name))
        except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
            raise ExecutionError(safe_detail(exc)) from exc

        if not matches:
            raise ExecutionError(
                "The forecasting job is not deployed in the connected Databricks workspace. "
                "An administrator needs to deploy the ForecastIQ bundle."
            )

        self._job_id = int(matches[0].job_id)
        return self._job_id

    def _monitor_job_status(self, databricks_run_id: int) -> tuple[JobStatus, str | None, float | None]:
        """Poll one Databricks run.

        Returns:
            Its `JobStatus`, a user-safe failure message (None unless it
            failed), and the run's duration in seconds once known.
        """
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
        """Read the completed run's summary.json back from the UC Volume.

        Falls back to MLflow when the file cannot be read: the engine logs
        the identical summary as a run artifact, so a volume-permission
        problem costs latency rather than the whole result.
        """
        payload = self._read_volume_json(record.summary_uri)
        if payload is not None:
            return payload
        return self._history.get_summary(record.run_id)

    def read_volume_text(self, uri: str) -> str | None:
        """A UC Volume file's contents as text, or None if unreadable.

        Public because it is the only way a caller outside this package
        (the Results page's dataset preview) can reach a file the *job*
        wrote: a UC Volume path is meaningless on the API host's own
        filesystem, so it has to be fetched through this workspace client.
        """
        try:
            response = self._workspace.files.download(uri)
            return response.contents.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - absence is normal, not an error
            return None

    def _read_volume_json(self, uri: str | None) -> dict[str, Any] | None:
        if not uri:
            return None
        try:
            response = self._workspace.files.download(uri)
            contents = response.contents.read()
        except Exception:  # noqa: BLE001 - absence is normal, not an error
            # Not written yet (the job has not reached that stage), or not
            # readable. Both resolve on a later poll; neither is worth
            # failing a status request over.
            return None

        try:
            return json.loads(contents)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_root(self, run_id: str) -> str:
        """One directory per run, holding its dataset, config and output."""
        return f"{self._settings.databricks_volumes_root.rstrip('/')}/runs/{run_id}"

    def _refresh(self, record: _DatabricksJobRecord) -> None:
        """Bring one record up to date with the workspace.

        Called from every read path rather than from a polling thread: a
        status is only interesting when someone asks for it, and this way
        a backend with a hundred historical runs does no background work.
        """
        if record.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            return
        if record.databricks_run_id is None:
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


def _enum_value(value: Any) -> str | None:
    """The string form of an SDK enum, or of a plain string.

    The SDK returns typed enums, but a mocked or future client may return
    raw strings; both must map identically.
    """
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
