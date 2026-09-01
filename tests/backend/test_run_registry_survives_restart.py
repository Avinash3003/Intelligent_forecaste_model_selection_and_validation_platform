"""A run still queued or booting compute must survive a backend restart.

`_DatabricksJobRecord` is documented as not surviving a restart, and
`get_run()` already restores a *finished* run from MLflow. But MLflow only
has a run once the engine's own tracking_pipeline has started inside the
job -- a run still PENDING or RUNNING at the moment a process restarts (a
new `--reload` cycle, a redeploy, a crash) falls in the gap between "the
in-memory record is gone" and "MLflow never heard of it yet". Without a
fallback, `get_run()` returns None and the API answers 404 -- exactly the
"Job run not available or not found" the user saw for a run that was, in
truth, still healthily RUNNING on Databricks the whole time.

Reproduced by submitting through one DatabricksRunner, then reading the
same run_id back through a *second*, freshly constructed DatabricksRunner
sharing the same fake workspace (so the UC Volume breadcrumb persists)
but starting with an empty in-memory `_jobs` dict -- the exact shape of a
process restart.
"""

from __future__ import annotations

import pytest

from app.orchestration.databricks_runner import DatabricksRunner
from app.orchestration.schemas import JobStatus
from test_databricks_runner import _FakeWorkspace, _request, dataset, settings  # noqa: F401


def test_a_still_running_run_is_recovered_after_a_restart(settings, dataset):
    workspace = _FakeWorkspace()
    first_runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = first_runner.submit(_request(dataset))

    # A new process: empty in-memory state, same fake workspace (the UC
    # Volume breadcrumb outlives the process that wrote it) and a fresh
    # MLflowHistoryStore against the same, still-empty tracking db.
    restarted_runner = DatabricksRunner(settings, workspace_client=workspace)

    listing = restarted_runner.get_run(run_id)

    assert listing is not None
    assert listing.run_id == run_id
    assert listing.job_status == JobStatus.RUNNING
    assert listing.databricks_run_id == 99


def test_a_run_with_no_registry_breadcrumb_still_reports_not_found(settings, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)

    assert runner.get_run("dbx-run-never-existed") is None
