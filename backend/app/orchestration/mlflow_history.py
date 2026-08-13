"""Persistent execution history, read from MLflow.

MLflow is already the platform's system of record for a finished run —
parameters, metrics, artifacts, registered models, and (since the tracking
layer opens its Parent Run before the first stage) the run's terminal
status and failure reason too. This module is the read side of that: it
reconstructs past executions from the tracking store so run history
survives a backend restart, which an in-memory job registry cannot.

The split it enforces is deliberate:

  * **Memory** owns *active* execution — RUNNING/PENDING jobs, the
    subprocess handle, cancellation. Those exist only while a run is
    executing and are meaningless after a restart.
  * **MLflow** owns *finished* execution — everything a completed or
    failed run left behind.

Nothing here is ever consulted for live status; a run still executing is
answered from memory. That is what keeps MLflow an experiment-tracking
store rather than a job scheduler.

Read-only by construction: this module never starts, ends, or mutates an
MLflow run. Writing is the engine's job alone, which is what keeps a
single writer and no duplicate source of truth.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.orchestration.schemas import ExecutionBackend, JobStatus, RunListing

logger = logging.getLogger(__name__)

# Mirrors forecast_engine/s12_tracking/run_summary.py. Duplicated as plain
# constants rather than imported because forecast_engine is a separate
# package with its own venv and is never importable from this process.
SUMMARY_ARTIFACT_PATH = "run/summary.json"
RUN_ID_TAG = "run_id"
DATASET_NAME_TAG = "dataset_name"
ERROR_TAG = "error"
FAILED_STAGE_TAG = "failed_stage"
# Written by the engine's own `begin()` — see forecast_engine/s12_tracking/
# run_summary.py for why these exist alongside the tags above.
STARTED_BY_USER_ID_TAG = "started_by_user_id"
STARTED_BY_DISPLAY_NAME_TAG = "started_by_display_name"
# Written by this module alone (`mark_cancelled`), never by the engine: a
# cancellation is requested from *outside* the process being cancelled, so
# there is no engine-side code path that could ever set these.
CANCELLED_BY_USER_ID_TAG = "cancelled_by_user_id"
CANCELLED_BY_DISPLAY_NAME_TAG = "cancelled_by_display_name"
CANCELLED_AT_TAG = "cancelled_at"

# MLflow's own RunInfo.status vocabulary -> this platform's JobStatus.
# RUNNING maps to FAILED deliberately: a *persisted* run still marked
# RUNNING is one whose process died without closing its run (a kill, a
# crash, a host restart). It is not live — anything genuinely live is in
# memory and never reaches this reader.
_STATUS_MAP = {
    "FINISHED": JobStatus.COMPLETED,
    "FAILED": JobStatus.FAILED,
    "KILLED": JobStatus.CANCELLED,
    "RUNNING": JobStatus.FAILED,
    "SCHEDULED": JobStatus.FAILED,
}

DEFAULT_HISTORY_LIMIT = 200


class MLflowHistoryStore:
    """Reads finished-run history from the configured MLflow store."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tracking_uri = settings.mlflow_tracking_uri_resolved
        self._experiment_name = settings.mlflow_experiment_name_resolved
        self._client: Any = None
        self._lock = threading.Lock()
        # Restored summaries are cached: on Databricks each one is a remote
        # artifact download, and the Results page re-reads the same run on
        # every group/horizon change.
        self._summary_cache: dict[str, dict[str, Any]] = {}

    def is_available(self) -> bool:
        """Whether history can be read. Never raises."""
        return self._get_client() is not None

    def list_runs(self, limit: int = DEFAULT_HISTORY_LIMIT) -> list[RunListing]:
        """Every finished run in the configured experiment, newest first.

        Returns an empty list — never raises — if MLflow is unreachable or
        the experiment does not exist yet: a history view that cannot reach
        its store should degrade to "no history", not break the page that
        also has to show live runs.
        """
        client = self._get_client()
        if client is None:
            return []

        try:
            experiment = client.get_experiment_by_name(self._experiment_name)
            if experiment is None:
                return []
            runs = client.search_runs(
                [experiment.experiment_id],
                max_results=limit,
                order_by=["attributes.start_time DESC"],
            )
        except Exception as exc:  # noqa: BLE001 - history is best-effort
            logger.warning("Could not read run history from MLflow: %s", exc)
            return []

        listings = []
        for run in runs:
            listing = self._to_listing(run)
            if listing is not None:
                listings.append(listing)
        return listings

    def get_summary(self, run_id: str) -> dict[str, Any] | None:
        """The consolidated run record for `run_id`, or None if absent.

        This is the exact `PipelineContext.summary()` payload the engine
        logged, so the caller can hand it straight to `result_mapper` — the
        same function the live path uses — instead of reassembling a run
        from individual artifacts through a second mapping.
        """
        with self._lock:
            cached = self._summary_cache.get(run_id)
        if cached is not None:
            return cached

        client = self._get_client()
        if client is None:
            return None

        run = self._find_run(client, run_id)
        if run is None:
            return None

        try:
            local_path = client.download_artifacts(run.info.run_id, SUMMARY_ARTIFACT_PATH)
            summary = json.loads(Path(local_path).read_text())
        except Exception as exc:  # noqa: BLE001
            # A failed run legitimately has no summary: it died before the
            # tracking stage could log one.
            logger.info("No run summary artifact for run '%s': %s", run_id, exc)
            return None

        with self._lock:
            self._summary_cache[run_id] = summary
        return summary

    def get_listing(self, run_id: str, with_stages: bool = False) -> RunListing | None:
        """One run's history entry, or None if it is not in the store.

        `with_stages` additionally loads the run's stage trail out of the
        summary artifact. It is off by default because that costs an
        artifact download — acceptable for a detail view opened once,
        wasteful on the status poll that runs every few seconds.
        """
        client = self._get_client()
        if client is None:
            return None
        run = self._find_run(client, run_id)
        if run is None:
            return None

        listing = self._to_listing(run)
        if listing is not None and with_stages:
            summary = self.get_summary(run_id)
            listing.stages = (summary or {}).get("stages") or []
        return listing

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_run(self, client: Any, run_id: str) -> Any:
        # Looked up by the *platform's* run id, held as a tag — MLflow's own
        # run id is a different identifier the rest of the platform never
        # uses.
        try:
            experiment = client.get_experiment_by_name(self._experiment_name)
            if experiment is None:
                return None
            runs = client.search_runs(
                [experiment.experiment_id],
                filter_string=f"tags.{RUN_ID_TAG} = '{run_id}'",
                max_results=1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not look up run '%s' in MLflow: %s", run_id, exc)
            return None
        return runs[0] if runs else None

    def _to_listing(self, run: Any) -> RunListing | None:
        tags = run.data.tags or {}
        platform_run_id = tags.get(RUN_ID_TAG)
        if not platform_run_id:
            # Not a run this platform submitted (e.g. logged directly from a
            # notebook against the same experiment). Skipped rather than
            # shown as a phantom deployment.
            return None

        status = _STATUS_MAP.get(run.info.status, JobStatus.FAILED)
        started_at = _iso_from_millis(run.info.start_time)
        completed_at = _iso_from_millis(run.info.end_time)

        duration = None
        if run.info.start_time and run.info.end_time:
            duration = max((run.info.end_time - run.info.start_time) / 1000.0, 0.0)

        error = tags.get(ERROR_TAG)
        if status is JobStatus.FAILED and not error and run.info.status == "RUNNING":
            error = "The run did not finish — its process ended without recording an outcome."

        return RunListing(
            run_id=platform_run_id,
            dataset_name=tags.get(DATASET_NAME_TAG) or None,
            job_status=status,
            execution_backend=ExecutionBackend(self._settings.execution_mode),
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            error=error,
            started_by=tags.get(STARTED_BY_DISPLAY_NAME_TAG) or None,
            cancelled_by=tags.get(CANCELLED_BY_DISPLAY_NAME_TAG) or None,
            # Stages live in the consolidated summary artifact, not in tags;
            # the history list deliberately does not download an artifact
            # per run just to render a table.
            stages=[],
        )

    def mark_cancelled(
        self,
        run_id: str,
        cancelled_by_user_id: str | None,
        cancelled_by_display_name: str | None,
        cancelled_at: str,
    ) -> bool:
        """Reconcile a cancelled run's MLflow Parent Run, if one exists.

        A run cancelled before its Parent Run could even be opened (still
        PENDING, or tracking disabled/unreachable) has nothing to reconcile
        here — that is a normal case, reported as `False`, not an error.

        Idempotent: safe to call more than once for the same run. Tags are
        simply re-set to the same values, and `set_terminated` is only
        attempted while MLflow still reports the run `RUNNING` — calling
        this again after that succeeds is a no-op beyond re-setting tags.

        Deliberately does not delete the MLflow run itself: `begin()` logs
        only `run_id`/`dataset_name`/`started_by_*` as tags, and nothing
        else is ever logged to a run before `track()` — the final stage a
        cancelled run by definition never reached. So a cancelled run's
        MLflow record is already exactly the minimal metadata Section 7
        (Feature 1.4) asks history to retain: marking it `KILLED` is
        sufficient, there is no larger artifact set to strip away first.
        """
        client = self._get_client()
        if client is None:
            return False

        run = self._find_run(client, run_id)
        if run is None:
            return False

        mlflow_run_id = run.info.run_id
        client.set_tag(mlflow_run_id, CANCELLED_BY_USER_ID_TAG, cancelled_by_user_id or "")
        client.set_tag(mlflow_run_id, CANCELLED_BY_DISPLAY_NAME_TAG, cancelled_by_display_name or "")
        client.set_tag(mlflow_run_id, CANCELLED_AT_TAG, cancelled_at)

        if run.info.status == "RUNNING":
            from mlflow.entities import RunStatus

            client.set_terminated(mlflow_run_id, status=RunStatus.to_string(RunStatus.KILLED))

        with self._lock:
            self._summary_cache.pop(run_id, None)
        return True

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is not None:
                return self._client
        try:
            from mlflow.tracking import MlflowClient
        except ImportError:
            logger.warning("mlflow is not installed; run history is unavailable.")
            return None

        try:
            client = MlflowClient(tracking_uri=self._tracking_uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not create an MLflow client for '%s': %s", self._tracking_uri, exc)
            return None

        with self._lock:
            self._client = client
        return client


def _iso_from_millis(value: int | None) -> str | None:
    if not value:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).isoformat(timespec="seconds")
