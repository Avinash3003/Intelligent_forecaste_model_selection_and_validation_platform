"""Databricks Runner (Section 6.14) — architecture only.

Implements the exact same `PipelineRunner` interface `LocalRunner` does, so
`PipelineExecutor` can select this backend by configuration alone with no
other code changing. What it does *not* do, deliberately, per this phase's
scope, is make a single real Azure Storage or Databricks Jobs API call —
every integration point below is a fully-specified method with the
signature, docstring and error contract a later phase implements against,
raising `DatabricksNotImplementedError` in the meantime.

The four placeholder methods mirror this Runner's real future control
flow exactly:

    submit()
        -> _upload_data_to_storage()   (dataset -> Azure Storage URI)
        -> _trigger_databricks_job()   (URI -> Databricks run id)
    get_status()
        -> _monitor_job_status()       (Databricks run id -> JobStatus)
    get_result()
        -> _retrieve_pipeline_results() (Databricks run id -> PipelineExecutionResult)

so that filling them in later is implementing four methods against an
already-correct shape, not redesigning this class.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config.settings import Settings
from app.orchestration.exceptions import DatabricksNotImplementedError, RunNotReadyError, UnknownRunError
from app.orchestration.mlflow_history import MLflowHistoryStore
from app.orchestration.runner_base import PipelineRunner
from app.orchestration.schemas import (
    ExecutionBackend,
    JobStatus,
    PipelineExecutionRequest,
    PipelineExecutionResult,
    RunListing,
)


@dataclass
class _DatabricksJobRecord:
    """In-memory record of one submission. A real implementation would
    persist `databricks_run_id` (and drop the in-memory dependency
    entirely) so status survives a backend restart — out of scope here.
    """

    run_id: str
    status: JobStatus
    started_at: str
    dataset_name: str | None = None
    databricks_run_id: str | None = None
    storage_uri: str | None = None
    error: str | None = None


class DatabricksRunner(PipelineRunner):
    """Placeholder architecture for executing the pipeline as a Databricks
    Job. See the module docstring for what is, and is not, implemented.
    """

    def __init__(self, settings: Settings, history: MLflowHistoryStore | None = None) -> None:
        self._settings = settings
        self._jobs: dict[str, _DatabricksJobRecord] = {}
        self._lock = threading.Lock()
        # Identical history source to LocalRunner. Retargeting to a
        # Databricks-managed tracking server is `MLFLOW_TRACKING_URI`
        # changing, not a different reader — which is what makes run
        # history behave the same in both execution modes.
        self._history = history or MLflowHistoryStore(settings)

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
            record.storage_uri = self._upload_data_to_storage(request)
            record.databricks_run_id = self._trigger_databricks_job(request, record.storage_uri)
        except DatabricksNotImplementedError as exc:
            record.status = JobStatus.FAILED
            record.error = str(exc)

        return run_id

    def get_status(self, run_id: str) -> JobStatus:
        record = self._require_job(run_id)
        if record.status in (JobStatus.PENDING, JobStatus.RUNNING) and record.databricks_run_id:
            try:
                record.status = self._monitor_job_status(record.databricks_run_id)
            except DatabricksNotImplementedError as exc:
                record.status = JobStatus.FAILED
                record.error = str(exc)
        return record.status

    def get_result(self, run_id: str) -> PipelineExecutionResult:
        record = self._require_job(run_id)
        if record.status in (JobStatus.PENDING, JobStatus.RUNNING):
            raise RunNotReadyError(f"Run '{run_id}' is still {record.status.value}; poll get_status() first.")

        if record.status is JobStatus.COMPLETED and record.databricks_run_id:
            try:
                return self._retrieve_pipeline_results(run_id, record.databricks_run_id)
            except DatabricksNotImplementedError as exc:
                record.status = JobStatus.FAILED
                record.error = str(exc)

        return PipelineExecutionResult(
            run_id=run_id,
            job_status=record.status,
            execution_backend=ExecutionBackend.DATABRICKS,
            started_at=record.started_at,
            error=record.error,
        )

    def list_runs(self) -> list[RunListing]:
        with self._lock:
            records = list(self._jobs.values())

        # No `stages`: a real implementation would read them back from the
        # completed job's summary artifact in `_retrieve_pipeline_results`,
        # which this phase does not implement.
        active = [
            RunListing(
                run_id=record.run_id,
                dataset_name=record.dataset_name,
                job_status=record.status,
                execution_backend=ExecutionBackend.DATABRICKS,
                started_at=record.started_at,
                error=record.error,
            )
            for record in records
        ]

        # Finished runs come from the tracking store, exactly as they do
        # locally — a Databricks job logs to MLflow through the same engine
        # code, so its history reads back identically.
        known = {listing.run_id for listing in active}
        merged = active + [listing for listing in self._history.list_runs() if listing.run_id not in known]
        merged.sort(key=lambda listing: listing.started_at or "", reverse=True)
        return merged

    def get_run(self, run_id: str) -> RunListing | None:
        with self._lock:
            record = self._jobs.get(run_id)
        if record is not None:
            return RunListing(
                run_id=record.run_id,
                dataset_name=record.dataset_name,
                job_status=record.status,
                execution_backend=ExecutionBackend.DATABRICKS,
                started_at=record.started_at,
                error=record.error,
            )
        return self._history.get_listing(run_id, with_stages=True)

    def cancel(self, run_id: str) -> bool:
        record = self._require_job(run_id)
        if record.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            return False
        record.status = JobStatus.CANCELLED
        return True

    def _require_job(self, run_id: str) -> _DatabricksJobRecord:
        with self._lock:
            record = self._jobs.get(run_id)
        if record is None:
            raise UnknownRunError(f"No run found for run_id '{run_id}'.")
        return record

    # ------------------------------------------------------------------
    # Placeholder integration points — Section 6.14 explicitly excludes
    # implementing these; a later Azure deployment phase fills them in.
    # ------------------------------------------------------------------

    def _upload_data_to_storage(self, request: PipelineExecutionRequest) -> str:
        """Upload the dataset to Azure Storage and return its URI.

        A real implementation authenticates against
        `settings.azure_storage_connection_string` and uploads
        `request.dataset_path`, returning an `abfss://` or `https://`
        blob URI the Databricks job can read from.
        """
        raise DatabricksNotImplementedError(
            "Azure Storage upload is not implemented in this phase (Section 6.14 scope)."
        )

    def _trigger_databricks_job(self, request: PipelineExecutionRequest, storage_uri: str) -> str:
        """Trigger the Databricks Job that runs `forecast_engine` against
        the uploaded dataset, and return the Databricks run id.

        A real implementation calls the Jobs API
        (`POST /api/2.2/jobs/run-now`) against
        `settings.databricks_host`/`settings.databricks_token`, passing
        `storage_uri` and `request.forecast_configuration` as job
        parameters.
        """
        raise DatabricksNotImplementedError(
            "Databricks Job triggering is not implemented in this phase (Section 6.14 scope)."
        )

    def _monitor_job_status(self, databricks_run_id: str) -> JobStatus:
        """Poll the Databricks Job's run state and translate it into this
        platform's `JobStatus` vocabulary.

        A real implementation calls `GET /api/2.2/jobs/runs/get` and maps
        Databricks' `life_cycle_state`/`result_state` onto
        PENDING/RUNNING/COMPLETED/FAILED/CANCELLED.
        """
        raise DatabricksNotImplementedError(
            "Databricks Job status monitoring is not implemented in this phase (Section 6.14 scope)."
        )

    def _retrieve_pipeline_results(self, run_id: str, databricks_run_id: str) -> PipelineExecutionResult:
        """Fetch the completed job's `PipelineContext.summary()` output
        (written by the same MLflow/forecast_engine logging Local
        execution uses) and map it through `result_mapper`, exactly as
        `LocalRunner` does.

        A real implementation reads the summary artifact back from Azure
        Storage or the Databricks Jobs API's run output, then calls
        `result_mapper.map_summary_to_result(...)` — the same function
        `LocalRunner` uses, so both backends produce an identical envelope.
        """
        raise DatabricksNotImplementedError(
            "Databricks result retrieval is not implemented in this phase (Section 6.14 scope)."
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
