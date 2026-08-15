"""Logs the whole run record as one artifact.

The context's summary is already a complete plain-JSON record, and the
backend already knows how to turn it into a result. Logging exactly that
object is what lets a past run be rebuilt from MLflow alone: the reader
downloads this file and reuses the same mapper the live path uses, instead
of reassembling the run from eight artifacts through a parallel mapping that
would have to be kept in step.

The granular artifacts remain the browsable view in the MLflow UI; this is
the machine-readable whole.
"""

from __future__ import annotations

# Path inside the run's artifact store. Both the engine (writer) and the
# backend (reader) import this constant rather than repeating the literal.
SUMMARY_ARTIFACT_PATH = "run/summary.json"

# Tag names the run-history view searches and reads runs by. `run_id` is
# the platform's own id (e.g. "fe-run-ab12cd34"), which is *not* MLflow's
# run id — history is looked up by the former.
#
# Run status is deliberately absent: MLflow's native `RunInfo.status`
# (RUNNING / FINISHED / FAILED / KILLED) already carries it, and a tag
# copy would be a second source of truth that can go stale.
RUN_ID_TAG = "run_id"
DATASET_NAME_TAG = "dataset_name"
ERROR_TAG = "error"
FAILED_STAGE_TAG = "failed_stage"

# Who submitted the run — set once, at `begin()`, alongside `run_id` and
# `dataset_name`, so even a run that dies before any other stage still
# carries who started it. `cancelled_by_*` has no engine-side counterpart:
# a cancellation is requested from outside this process (the backend, not
# the pipeline it is cancelling), so those tags are written directly by the
# backend's own MLflow client, not through this module.
STARTED_BY_USER_ID_TAG = "started_by_user_id"
STARTED_BY_DISPLAY_NAME_TAG = "started_by_display_name"
