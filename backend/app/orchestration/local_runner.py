"""Runs the forecast engine as a local subprocess.

The engine has its own venv and heavy dependencies (xgboost, statsmodels,
mlflow), so it is launched as a separate process rather than imported.
Submit returns immediately and the run happens on a background thread.

Job state is split by lifetime:
  - in memory: only active PENDING/RUNNING jobs and the subprocess handle.
  - in MLflow: every finished run, which is what survives a restart.

A finished job leaves memory only once MLflow confirms it, so a run is never
lost because tracking was unreachable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.orchestration.exceptions import RunNotReadyError, RunnerConfigurationError, UnknownRunError
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


@dataclass
class _JobRecord:
    """In-memory state for one run, guarded by LocalRunner._lock."""

    run_id: str
    status: JobStatus
    started_at: str
    dataset_name: str | None = None
    process: subprocess.Popen | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    # Path the engine writes its stage trail to after every transition
    # (Section "Live execution tracking"). Set once, at submit time; read
    # on demand while the job is still active, never written by this
    # process — the engine subprocess is the sole writer.
    live_status_path: Path | None = None
    # The per-run temp directory holding the dataset config, live status,
    # summary and stderr log — set once, at execute time, so cancellation
    # can delete the whole thing in one shot rather than naming each file.
    work_dir: Path | None = None
    started_by_user_id: str | None = None
    started_by_display_name: str | None = None
    cancelled_by_user_id: str | None = None
    cancelled_by_display_name: str | None = None


class LocalRunner(PipelineRunner):
    """Runs `forecast_engine.run_pipeline` as a local subprocess."""

    def __init__(self, settings: Settings, history: MLflowHistoryStore | None = None) -> None:
        self._settings = settings
        self._jobs: dict[str, _JobRecord] = {}
        self._lock = threading.Lock()
        self._history = history or MLflowHistoryStore(settings)

        python_path = settings.forecast_engine_python_path
        if not python_path.exists():
            # Fails fast at construction rather than on first submit, so a
            # misconfigured deployment is caught at backend startup.
            raise RunnerConfigurationError(
                f"Configured forecast_engine interpreter '{python_path}' does not exist. "
                f"Check FORECAST_ENGINE_ROOT / FORECAST_ENGINE_PYTHON."
            )

    def submit(self, request: PipelineExecutionRequest) -> str:
        run_id = request.run_id or f"fe-run-{uuid.uuid4().hex[:12]}"
        record = _JobRecord(
            run_id=run_id,
            status=JobStatus.PENDING,
            started_at=_now_iso(),
            # Falls back to the staged file's own name so a run submitted
            # without an explicit dataset name is still identifiable.
            dataset_name=request.dataset_name or Path(request.dataset_path).name,
            started_by_user_id=request.started_by_user_id,
            started_by_display_name=request.started_by_display_name,
        )

        with self._lock:
            self._jobs[run_id] = record

        # Daemon thread: if the backend process exits, in-flight local runs
        # exit with it rather than becoming orphaned background work.
        thread = threading.Thread(target=self._execute, args=(run_id, request), daemon=True)
        thread.start()
        return run_id

    def get_status(self, run_id: str) -> JobStatus:
        record = self._find_job(run_id)
        if record is not None:
            return record.status

        # Not active — it either finished before this process started, or
        # finished and was already handed over to MLflow.
        listing = self._history.get_listing(run_id)
        if listing is None:
            raise UnknownRunError(f"No run found for run_id '{run_id}'.")
        return listing.job_status

    def get_result(self, run_id: str) -> PipelineExecutionResult:
        record = self._find_job(run_id)
        if record is None:
            return self._restore_result(run_id)

        if record.status in (JobStatus.PENDING, JobStatus.RUNNING):
            raise RunNotReadyError(f"Run '{run_id}' is still {record.status.value}; poll get_status() first.")

        if record.status is not JobStatus.COMPLETED or record.summary is None:
            return PipelineExecutionResult(
                run_id=run_id,
                job_status=record.status,
                execution_backend=ExecutionBackend.LOCAL,
                started_at=record.started_at,
                completed_at=record.completed_at,
                duration_seconds=record.duration_seconds,
                error=record.error,
            )

        return map_summary_to_result(
            run_id=run_id,
            execution_backend=ExecutionBackend.LOCAL,
            job_status=record.status,
            summary=record.summary,
            started_at=record.started_at,
            completed_at=record.completed_at,
            duration_seconds=record.duration_seconds,
        )

    def _restore_result(self, run_id: str) -> PipelineExecutionResult:
        """Rebuild a finished run from MLflow.

        Uses the same summary payload and the same mapper as a live run, so
        both produce one identical object.
        """
        listing = self._history.get_listing(run_id)
        if listing is None:
            raise UnknownRunError(f"No run found for run_id '{run_id}'.")

        summary = self._history.get_summary(run_id) if listing.job_status is JobStatus.COMPLETED else None
        if summary is None:
            # A failed run has no summary — it died before the tracking
            # stage could log one. The envelope still carries why.
            return PipelineExecutionResult(
                run_id=run_id,
                job_status=listing.job_status,
                execution_backend=listing.execution_backend,
                started_at=listing.started_at,
                completed_at=listing.completed_at,
                duration_seconds=listing.duration_seconds,
                error=listing.error,
            )

        return map_summary_to_result(
            run_id=run_id,
            execution_backend=listing.execution_backend,
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

        active = [self._to_listing(record) for record in records]

        # In-memory entries win over persisted ones for the same id: a job
        # this process is running is more current than whatever MLflow has
        # recorded for it so far.
        known = {listing.run_id for listing in active}
        merged = active + [listing for listing in self._history.list_runs() if listing.run_id not in known]

        # Most recent first: a history view's newest entry is the one a user
        # just submitted and is waiting on.
        merged.sort(key=lambda listing: listing.started_at or "", reverse=True)
        return merged

    def get_run(self, run_id: str) -> RunListing | None:
        record = self._find_job(run_id)
        if record is not None:
            return self._to_listing(record)
        return self._history.get_listing(run_id, with_stages=True)

    def _to_listing(self, record: _JobRecord) -> RunListing:
        return RunListing(
            run_id=record.run_id,
            dataset_name=record.dataset_name,
            job_status=record.status,
            execution_backend=ExecutionBackend.LOCAL,
            started_at=record.started_at,
            completed_at=record.completed_at,
            duration_seconds=record.duration_seconds,
            error=record.error,
            started_by=record.started_by_display_name,
            cancelled_by=record.cancelled_by_display_name,
            stages=self._current_stages(record),
        )

    def _current_stages(self, record: _JobRecord) -> list[dict[str, Any]]:
        """The job's real stage trail.

        Finished jobs read from the final summary; active ones from the
        engine's live-status file, rewritten after every stage transition.
        """
        if record.summary is not None:
            return record.summary.get("stages") or []

        if record.live_status_path is None:
            return []
        try:
            payload = json.loads(record.live_status_path.read_text())
        except (OSError, ValueError):
            # Not written yet (process hasn't reached stage 1) or caught
            # mid-write despite the atomic rename — both resolve on the
            # next poll, a few seconds later.
            return []
        return payload.get("stages") or []

    # Grace period after SIGTERM before escalating to SIGKILL. Bounded and
    # short: cleanup below must not start deleting a run's storage while
    # the subprocess could still be mid-write to one of those same paths,
    # but a cancel request is also a user waiting on this call to return.
    _TERMINATE_GRACE_SECONDS = 5.0

    def cancel(
        self,
        run_id: str,
        cancelled_by_user_id: str | None = None,
        cancelled_by_display_name: str | None = None,
    ) -> CancellationOutcome:
        record = self._require_job(run_id)
        with self._lock:
            if record.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                # Already terminal — nothing to do, and nothing to report as
                # a failure. A second cancel() on the same run lands here.
                return CancellationOutcome(cancelled=False)
            record.status = JobStatus.CANCELLED
            record.completed_at = _now_iso()
            record.cancelled_by_user_id = cancelled_by_user_id
            record.cancelled_by_display_name = cancelled_by_display_name
            process = record.process
            work_dir = record.work_dir

        # Terminated outside the lock: killing a process is a syscall, not
        # a state mutation, and should never be made while holding it. A
        # forced kill is not something the subprocess's own `except
        # Exception` can react to — cleanup below happens from this side,
        # not by trusting the subprocess to clean up after itself.
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self._TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                # Still alive after a polite SIGTERM — stop waiting and
                # force it, so cleanup below never races a process that is
                # still writing to the very paths it is about to delete.
                process.kill()
                process.wait(timeout=self._TERMINATE_GRACE_SECONDS)

        cleanup_errors = self._cleanup_run_storage(run_id, work_dir)

        try:
            self._history.mark_cancelled(
                run_id, cancelled_by_user_id, cancelled_by_display_name, record.completed_at
            )
        except Exception as exc:  # noqa: BLE001 - report, never raise out of cancel()
            cleanup_errors.append(f"MLflow run lifecycle: {exc}")

        return CancellationOutcome(cancelled=True, cleanup_errors=cleanup_errors)

    def _cleanup_run_storage(self, run_id: str, work_dir: Path | None) -> list[str]:
        """Delete everything this run wrote, scoped strictly to its run_id.

        Idempotent — a missing path counts as already clean. Never touches
        uploads/, which is keyed by file_id and may serve several runs.
        Returns the locations that could not be removed; empty means all clean.
        """
        errors: list[str] = []

        if work_dir is not None:
            try:
                shutil.rmtree(work_dir)
            except FileNotFoundError:
                pass
            except OSError as exc:
                errors.append(f"run workspace ({work_dir}): {exc}")

        # Mirrors forecast_engine's own default storage roots (Sections
        # curated_storage/model_storage/forecast_export/artifacts_mirror in
        # config/pipeline_config.py) — LocalRunner does not override them,
        # so this must reconstruct the same paths the engine actually wrote.
        engine_root = Path(self._settings.forecast_engine_root).resolve().parent
        for label, path in (
            ("curated dataset", engine_root / "curated" / run_id),
            ("trained models", engine_root / "models" / run_id),
            ("mirrored artifacts", engine_root / "artifacts" / run_id),
        ):
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                errors.append(f"{label} ({path}): {exc}")

        forecast_csv = engine_root / "forecasts" / f"{run_id}_forecast.csv"
        try:
            forecast_csv.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            errors.append(f"forecast export ({forecast_csv}): {exc}")

        return errors

    def _find_job(self, run_id: str) -> _JobRecord | None:
        with self._lock:
            return self._jobs.get(run_id)

    def _require_job(self, run_id: str) -> _JobRecord:
        record = self._find_job(run_id)
        if record is None:
            raise UnknownRunError(f"No run found for run_id '{run_id}'.")
        return record

    def _hand_over_to_history(self, run_id: str) -> None:
        """Drop a finished job from memory once MLflow holds its outcome.

        Waits for MLflow's end_time, not merely for a row to exist. The
        parent run is opened before stage 1, so a row appears immediately;
        a job killed uncatchably (OOM, segfault) never closes it. Handing
        over on row-existence alone would replace the real in-memory error
        and stage trail with history's generic "did not finish".
        """
        try:
            listing = self._history.get_listing(run_id)
            persisted = listing is not None and listing.completed_at is not None
        except Exception:  # noqa: BLE001 - retention is the safe fallback
            persisted = False

        if not persisted:
            return

        with self._lock:
            self._jobs.pop(run_id, None)

    def _execute(self, run_id: str, request: PipelineExecutionRequest) -> None:
        """Runs on a background thread — the actual subprocess lifecycle."""
        record = self._require_job(run_id)
        with self._lock:
            record.status = JobStatus.RUNNING

        work_dir = Path(tempfile.mkdtemp(prefix=f"forecast-run-{run_id}-"))
        config_path = work_dir / "forecast_configuration.json"
        summary_path = work_dir / "summary.json"
        live_status_path = work_dir / "live_status.json"

        with self._lock:
            record.live_status_path = live_status_path
            record.work_dir = work_dir

        try:
            config_path.write_text(json.dumps(request.forecast_configuration))

            command = [
                str(self._settings.forecast_engine_python_path),
                "-m",
                "forecast_engine.run_pipeline",
                "--dataset",
                request.dataset_path,
                "--config",
                str(config_path),
                "--run-id",
                run_id,
                "--summary-out",
                str(summary_path),
                "--live-status-out",
                str(live_status_path),
            ]
            if request.selected_models:
                command += ["--models", *request.selected_models]
            if request.fallback_model:
                command += ["--fallback-model", request.fallback_model]
            if request.derived_features is not None:
                # `is not None`, not truthy: an explicit empty list means
                # "select none" and must still reach the engine as such,
                # not be treated as "not specified" and silently default.
                command += ["--derived-features", *request.derived_features]
            if request.horizon is not None:
                command += ["--horizon", str(request.horizon)]
            if record.dataset_name:
                # Recorded with the run in MLflow, so run history shows the
                # name a user uploaded rather than the staged file's.
                command += ["--dataset-name", record.dataset_name]
            if record.started_by_user_id:
                command += ["--started-by-user-id", record.started_by_user_id]
            if record.started_by_display_name:
                command += ["--started-by-display-name", record.started_by_display_name]

            process = subprocess.Popen(
                command,
                # `-m forecast_engine.run_pipeline` resolves the top-level
                # package from the current working directory, so this must
                # be forecast_engine's *parent* — the project root — not
                # the forecast_engine directory itself.
                cwd=str(Path(self._settings.forecast_engine_root).resolve().parent),
                # Explicit env, not the default (inherit-only) behaviour:
                # forecast_engine's Azure OpenAI / MLflow config is read
                # from *this* subprocess's environment, and the only place
                # those values exist after being loaded from backend/.env
                # is `Settings` — they are never auto-exported into this
                # process's own `os.environ`. `subprocess_env()` forwards
                # them explicitly so editing `backend/.env` alone is enough.
                env=self._settings.subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            with self._lock:
                # A cancel() that raced in before the process existed is
                # honoured immediately rather than letting the run start.
                raced_cancel = record.status is JobStatus.CANCELLED
                if not raced_cancel:
                    record.process = process

            if raced_cancel:
                # Waited on outside the lock, same as cancel() below: a
                # syscall must never run while holding it. Without this
                # wait the child was left unreaped — a zombie until the
                # whole backend process eventually exits.
                process.terminate()
                try:
                    process.wait(timeout=self._TERMINATE_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=self._TERMINATE_GRACE_SECONDS)
                return

            _, stderr = process.communicate(timeout=self._settings.job_timeout_seconds)

            # Persisted unconditionally, not just on failure: `stderr` only
            # ever lived in this local variable, and `record.error` keeps
            # just the last 500 characters. An uncatchable kill (OOM, a
            # native segfault) skips every other error-reporting path in
            # this method, so this file is the only place a future crash's
            # actual output survives long enough to be inspected.
            if stderr:
                try:
                    (work_dir / "stderr.log").write_text(stderr)
                except OSError:
                    pass

            with self._lock:
                if record.status is JobStatus.CANCELLED:
                    return  # cancel() already finalized this record

                if process.returncode != 0:
                    record.status = JobStatus.FAILED
                    record.error = _describe_failure(process.returncode, stderr)
                else:
                    record.summary = json.loads(summary_path.read_text())
                    record.status = JobStatus.COMPLETED

        except subprocess.TimeoutExpired:
            process.kill()
            # Reaped here, not left for the OS: an unreaped child stays a
            # zombie in the process table until this backend process itself
            # exits, and a deployment that regularly times out runs would
            # accumulate them for as long as it stays up.
            try:
                process.wait(timeout=self._TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
            with self._lock:
                record.status = JobStatus.FAILED
                record.error = f"Execution exceeded the configured timeout of {self._settings.job_timeout_seconds}s."
        except Exception as exc:  # noqa: BLE001 - a run's own failure must never crash the backend process
            with self._lock:
                record.status = JobStatus.FAILED
                record.error = f"Local execution failed: {type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                # A CANCELLED record was already finalized by cancel() —
                # including its own `completed_at` — before this subprocess
                # actually exited; overwriting it here with "now" would
                # quietly drift the cancellation timestamp later by however
                # long the process took to die after being signalled.
                if record.status is not JobStatus.CANCELLED:
                    record.completed_at = _now_iso()
                if record.started_at and record.completed_at:
                    record.duration_seconds = (
                        datetime.fromisoformat(record.completed_at) - datetime.fromisoformat(record.started_at)
                    ).total_seconds()

            # Outside the lock: this queries the tracking store, and holding
            # the registry lock across network/disk I/O would stall every
            # status poll for the duration.
            self._hand_over_to_history(run_id)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Turn a failed subprocess's stderr into the one line worth showing a user
def _describe_failure(return_code: int, stderr: str) -> str:
    # The engine prints its own "Forecast Engine failed: <reason>" line for
    # every handled error. Dependency warnings (MLflow type hints, CloudPickle
    # notices) routinely surround it, so surfacing raw stderr buries the
    # actual cause behind noise the user can do nothing about.
    for line in reversed((stderr or "").splitlines()):
        if line.startswith("Forecast Engine failed:"):
            return line[len("Forecast Engine failed:") :].strip()

    tail = (stderr or "").strip()[-500:]
    return f"forecast_engine exited with code {return_code}." + (f" {tail}" if tail else "")
