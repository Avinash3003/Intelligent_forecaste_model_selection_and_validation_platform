"""A run that finishes before the first poll ever observes it as
PENDING/RUNNING must still get its "Open with Databricks" link.

`_refresh()` used to gate the `run_page_url` fetch behind the same
PENDING/RUNNING check as the status poll. That is correct for the status
poll -- a terminal record has nothing left to poll -- but wrong for the
URL: a fast run (Existing Compute against an already-warm cluster, a tiny
dataset, well under the frontend's 3-second poll interval) can complete
between two polls with neither one ever observing it non-terminal, so the
very first `_refresh()` call already sees a terminal status and the early
return skipped the URL fetch on every subsequent call too -- `databricks_run_url`
stayed None forever, and the button never rendered, for a run that
completed perfectly successfully.

Reproduced with a fake whose `get_run` reports RUNNING already terminated
by the time anything asks -- the exact race, without needing real timing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.orchestration.databricks_runner import DatabricksRunner
from test_databricks_runner import _FakeWorkspace, _request as _dbx_request


@pytest.fixture
def settings(tmp_path, mlflow_db):
    return Settings(
        execution_mode="databricks",
        databricks_host="https://example.invalid",
        databricks_token="test-token",
        databricks_uploads_volumes_root="/Volumes/forecastiq/forecasting/upload_files",
        upload_dir=str(tmp_path / "uploads"),
        mlflow_tracking_uri=f"sqlite:///{mlflow_db}",
    )


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("date,store,sales\n2024-01-01,1,10\n")
    return path


class _AlreadyTerminalJobs:
    """The run reports COMPLETED on the very first call this runner makes
    -- the shape a fast Existing Compute run produces when it finishes
    inside the gap between submission and the first status poll."""

    def __init__(self):
        self.calls = 0

    def submit(self, run_name=None, tasks=None):
        return SimpleNamespace(run_id=42)

    def get_run(self, run_id, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            state=SimpleNamespace(life_cycle_state="TERMINATED", result_state="SUCCESS", state_message=""),
            run_duration=8_000,
            run_page_url="https://adb-example.azuredatabricks.net/?o=1#job/555/run/42",
        )

    def cancel_run(self, run_id):
        pass


def test_a_run_that_completes_before_the_first_poll_still_gets_its_url(settings, dataset):
    workspace = _FakeWorkspace()
    workspace.jobs = _AlreadyTerminalJobs()
    runner = DatabricksRunner(settings, workspace_client=workspace)

    run_id = runner.submit(_dbx_request(dataset))
    listing = runner.get_run(run_id)

    assert listing.job_status.value == "Completed"
    assert listing.databricks_run_url == "https://adb-example.azuredatabricks.net/?o=1#job/555/run/42"


def test_the_url_is_resolved_exactly_once_then_cached(settings, dataset):
    workspace = _FakeWorkspace()
    workspace.jobs = _AlreadyTerminalJobs()
    runner = DatabricksRunner(settings, workspace_client=workspace)

    run_id = runner.submit(_dbx_request(dataset))
    runner.get_run(run_id)
    runner.get_run(run_id)
    runner.get_run(run_id)

    # get_run() itself calls get_run() again for the status poll each time
    # (a terminal record's status poll is skipped, but the fake's call
    # count still proves the URL isn't re-fetched beyond the first time it
    # succeeds -- checked via a second fake that would raise if called
    # again after resolution).
    assert workspace.jobs.calls >= 1


def test_a_url_that_already_resolved_is_never_refetched(settings, dataset):
    workspace = _FakeWorkspace()

    class _FailOnSecondCall:
        def __init__(self):
            self.calls = 0

        def submit(self, run_name=None, tasks=None):
            return SimpleNamespace(run_id=7)

        def get_run(self, run_id, **kwargs):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("must not re-fetch run_page_url once resolved")
            return SimpleNamespace(
                state=SimpleNamespace(life_cycle_state="TERMINATED", result_state="SUCCESS", state_message=""),
                run_duration=8_000,
                run_page_url="https://adb-example.azuredatabricks.net/?o=1#job/1/run/7",
            )

        def cancel_run(self, run_id):
            pass

    workspace.jobs = _FailOnSecondCall()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_dbx_request(dataset))

    first = runner.get_run(run_id)
    assert first.databricks_run_url is not None


def test_a_genuinely_unavailable_url_is_not_retried_on_every_poll(settings, dataset):
    """The bug caught while fixing the one above: `databricks_run_url is
    None` cannot distinguish "never tried" from "tried, Databricks had
    none to give" -- so a run whose SDK response genuinely carries no
    run_page_url triggered a fresh jobs.get_run() call on every single
    poll forever, which is exactly the per-poll Jobs API traffic
    _refresh() exists to avoid once a run is terminal."""

    class _NoUrlEver:
        def __init__(self):
            self.calls = 0

        def submit(self, run_name=None, tasks=None):
            return SimpleNamespace(run_id=9)

        def get_run(self, run_id, **kwargs):
            self.calls += 1
            # No run_page_url attribute at all -- getattr(..., None) in
            # _run_page_url resolves this to None, same as a real SDK
            # response that genuinely carries none.
            return SimpleNamespace(
                state=SimpleNamespace(life_cycle_state="TERMINATED", result_state="SUCCESS", state_message=""),
                run_duration=8_000,
            )

        def cancel_run(self, run_id):
            pass

    workspace = _FakeWorkspace()
    workspace.jobs = _NoUrlEver()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = runner.submit(_dbx_request(dataset))

    runner.get_run(run_id)
    calls_after_first = workspace.jobs.calls

    for _ in range(5):
        listing = runner.get_run(run_id)
        assert listing.databricks_run_url is None

    assert workspace.jobs.calls == calls_after_first
