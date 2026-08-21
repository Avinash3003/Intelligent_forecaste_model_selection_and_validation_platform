"""Reads finished-run history back from MLflow.

Memory owns active runs; MLflow owns finished ones. That is what lets run
history survive a backend restart, and it keeps MLflow a tracking store
rather than a job scheduler — live status is never read from here.

Read-only apart from mark_cancelled: the engine is the only writer.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
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

# `max_results` is a PAGE SIZE, not a total, and a Databricks-hosted
# tracking server picks its own page size regardless of what is asked for —
# observed returning 1 run on the first page and 2 on the second for a
# three-run experiment. A single `search_runs` call therefore returns an
# arbitrary prefix of the history, so every caller must follow `.token`
# until it is empty. Not following it is invisible on a file-backed store
# (which answers in one page) and silently hides every older run in cloud
# mode — the whole dataset dropdown collapses to the most recent run.
#
# The page loop is bounded as well as token-driven: a server that kept
# handing back a token would otherwise spin forever. 200 pages is far more
# than DEFAULT_HISTORY_LIMIT can consume even at one run per page.
_MAX_SEARCH_PAGES = 200

# How long "this run has no summary artifact" is remembered. A *positive*
# summary is cached forever — a finished run's artifact never changes — but
# a negative one must expire: a run can legitimately have no summary yet
# (still executing, or died before the tracking stage) and then gain one.
# Without this, every caller re-attempted the same failing remote download
# on every request, which is most of what made estimation slow.
MISSING_SUMMARY_TTL_SECONDS = 120.0


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
        # run_id -> monotonic deadline until which "no summary" is taken on
        # trust instead of re-downloaded. See MISSING_SUMMARY_TTL_SECONDS.
        self._missing_summaries: dict[str, float] = {}

    def is_available(self) -> bool:
        """Whether history can be read. Never raises."""
        return self._get_client() is not None

    def list_runs(self, limit: int = DEFAULT_HISTORY_LIMIT) -> list[RunListing]:
        """Every finished run, newest first.

        Returns an empty list rather than raising when MLflow is
        unreachable, so history degrades to "none" instead of breaking the page.
        """
        client = self._get_client()
        if client is None:
            return []

        try:
            experiment = client.get_experiment_by_name(self._experiment_name)
            if experiment is None:
                return []
            runs = self._search_runs(
                client,
                experiment.experiment_id,
                limit=limit,
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
        """The engine's own summary payload for a run, or None.

        Handed straight to result_mapper, the same function the live path
        uses, so a restored run needs no second mapping.
        """
        with self._lock:
            cached = self._summary_cache.get(run_id)
            if cached is None and self._summary_is_known_missing(run_id):
                return None
        if cached is not None:
            return cached

        client = self._get_client()
        if client is None:
            return None

        run = self._find_run(client, run_id)
        if run is None:
            self._remember_missing_summary(run_id)
            return None

        try:
            local_path = client.download_artifacts(run.info.run_id, SUMMARY_ARTIFACT_PATH)
            summary = json.loads(Path(local_path).read_text())
        except Exception as exc:  # noqa: BLE001
            # A failed run legitimately has no summary: it died before the
            # tracking stage could log one.
            logger.info("No run summary artifact for run '%s': %s", run_id, exc)
            self._remember_missing_summary(run_id)
            return None

        with self._lock:
            self._summary_cache[run_id] = summary
            self._missing_summaries.pop(run_id, None)
        return summary

    # Caller must hold self._lock.
    def _summary_is_known_missing(self, run_id: str) -> bool:
        deadline = self._missing_summaries.get(run_id)
        if deadline is None:
            return False
        if time.monotonic() >= deadline:
            del self._missing_summaries[run_id]
            return False
        return True

    def _remember_missing_summary(self, run_id: str) -> None:
        with self._lock:
            self._missing_summaries[run_id] = time.monotonic() + MISSING_SUMMARY_TTL_SECONDS

    def get_listing(self, run_id: str, with_stages: bool = False) -> RunListing | None:
        """One run's history entry, or None.

        with_stages also loads the stage trail, which costs an artifact
        download — fine for a detail view, wasteful on a status poll.
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
            # Paged for the same reason list_runs is, and it matters more
            # here: a miss is not "one run missing from a table" but "this
            # run has no data", which is what the Results page renders when
            # the summary cannot be found. A filtered search still answers
            # in pages, so an empty first page carrying a token is a match
            # not yet reached — not an absent run.
            runs = self._search_runs(
                client,
                experiment.experiment_id,
                limit=1,
                filter_string=f"tags.{RUN_ID_TAG} = '{run_id}'",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not look up run '%s' in MLflow: %s", run_id, exc)
            return None
        return runs[0] if runs else None

    @staticmethod
    def _search_runs(
        client: Any,
        experiment_id: str,
        *,
        limit: int,
        filter_string: str = "",
        order_by: list[str] | None = None,
    ) -> list[Any]:
        """`client.search_runs`, following pagination to `limit` runs.

        See `_MAX_SEARCH_PAGES` for why a single call is not enough against
        a Databricks-hosted tracking server. Ordering is preserved: pages
        arrive in the server's own order and are concatenated in sequence.
        """
        collected: list[Any] = []
        token: str | None = None

        for _ in range(_MAX_SEARCH_PAGES):
            page = client.search_runs(
                [experiment_id],
                filter_string=filter_string,
                max_results=limit - len(collected),
                order_by=order_by,
                page_token=token,
            )
            collected.extend(page)
            if len(collected) >= limit:
                break
            # A PagedList carries `.token`; a plain list (a stub, or an SDK
            # that does not page) has none, which ends the loop after one
            # call exactly as the old single-call behaviour did.
            token = getattr(page, "token", None)
            if not token:
                break

        return collected[:limit]

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
        """Mark a cancelled run KILLED in MLflow, if it ever opened a run.

        Returns False (not an error) when there was nothing to reconcile —
        cancelled while still PENDING, or tracking unreachable.

        Idempotent. The MLflow run is not deleted: nothing but a few tags is
        logged before the final tracking stage, which a cancelled run never
        reaches, so KILLED already leaves the minimal record history needs.
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
            self._missing_summaries.pop(run_id, None)
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

        self._ensure_databricks_credentials_in_environment()

        try:
            client = MlflowClient(tracking_uri=self._tracking_uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not create an MLflow client for '%s': %s", self._tracking_uri, exc)
            return None

        with self._lock:
            self._client = client
        return client

    def _ensure_databricks_credentials_in_environment(self) -> None:
        """Make this backend's own Databricks credentials visible to MLflow.

        A `tracking_uri` of "databricks" authenticates by reading
        `DATABRICKS_HOST`/`DATABRICKS_CLIENT_ID`/`DATABRICKS_CLIENT_SECRET`/
        `DATABRICKS_TOKEN` from the process environment itself — it never
        reads this app's Settings object, which is how DatabricksRunner's
        WorkspaceClient authenticates instead (constructed explicitly from
        Settings fields). The two only agree automatically when Settings
        was populated from real environment variables with exactly those
        names; if this deployment's `.env` file provided the credentials
        (pydantic-settings loads a `.env` file into the Settings object
        without exporting it to `os.environ`), job submission still works
        but MLflow's independent credential lookup finds nothing and
        silently reports history as unavailable — which reads to a user as
        "my past runs disappeared" with no error anywhere.

        `setdefault` only: an operator's own real environment variable is
        never overridden, this only fills a gap.
        """
        if not self._tracking_uri.strip().lower().startswith("databricks"):
            return

        for env_name, value in (
            ("DATABRICKS_HOST", self._settings.databricks_host),
            ("DATABRICKS_CLIENT_ID", self._settings.databricks_client_id),
            ("DATABRICKS_CLIENT_SECRET", self._settings.databricks_client_secret),
            ("DATABRICKS_TOKEN", self._settings.databricks_token),
        ):
            if value:
                os.environ.setdefault(env_name, value)


def _iso_from_millis(value: int | None) -> str | None:
    if not value:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).isoformat(timespec="seconds")
