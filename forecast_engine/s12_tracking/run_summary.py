"""The consolidated run-record artifact — the contract between the engine
and any consumer that needs to rebuild a past run.

`PipelineContext.summary()` is already the complete, plain-JSON record of
one execution, and the backend's `result_mapper` already knows how to turn
it into the standardized result envelope. Logging exactly that object, as
one artifact, is what lets a completed run be reconstructed later from
MLflow alone: the reader downloads this file and reuses the same mapper
the live path uses, rather than reassembling the run from eight separate
artifacts through a second, parallel mapping that would have to be kept in
step with the first.

The granular artifacts (`forecast/`, `drift/`, `insights/`, …) are
unaffected and remain the browsable view in the MLflow UI. This one is the
machine-readable whole.
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
