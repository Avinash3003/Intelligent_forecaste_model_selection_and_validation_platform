"""Every Databricks run is a shared enterprise resource, not a personal one.

ForecastIQ operates inside one shared Databricks workspace: the same three
role groups (ForecastIQ-Admins/-DataScientists/-Analysts) that see the
platform's Unity Catalog data and its persistent dev job must also see
each run's job/run history in Databricks — regardless of which user
submitted it. `jobs.submit()` creates a brand-new one-time-run job for
every run (there is no single persistent job the app triggers by id — see
databricks/databricks.yml), so sharing has to happen per submission, not
once at deploy time. This is done by passing an access_control_list
straight into the same submit() call, atomically, rather than a
submit-then-separately-share pattern that would leave a race window (or a
crash window) where a just-submitted run is temporarily unshared.

Sharing is an enhancement, never a gate: a workspace where the groups
don't exist, or a transient failure setting the ACL, must never stop a
forecast from running.
"""

from __future__ import annotations

import pytest

from app.config.settings import Settings
from app.orchestration.databricks_runner import DatabricksRunner
from test_databricks_runner import _FakeWorkspace, _request


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


def test_every_run_is_submitted_with_the_three_role_groups_shared(settings, dataset):
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)

    runner.submit(_request(dataset))

    acl = workspace.jobs.submit_calls[0]["access_control_list"]
    assert acl is not None
    groups = {entry.group_name: entry.permission_level.value for entry in acl}
    assert groups == {
        "ForecastIQ-Admins": "CAN_MANAGE",
        "ForecastIQ-DataScientists": "CAN_VIEW",
        "ForecastIQ-Analysts": "CAN_VIEW",
    }


def test_a_group_left_unset_is_simply_not_shared_with(settings, dataset):
    settings = settings.model_copy(update={"databricks_analysts_group": None})
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)

    runner.submit(_request(dataset))

    acl = workspace.jobs.submit_calls[0]["access_control_list"]
    groups = {entry.group_name for entry in acl}
    assert groups == {"ForecastIQ-Admins", "ForecastIQ-DataScientists"}


def test_no_groups_configured_submits_with_no_acl_at_all(settings, dataset):
    settings = settings.model_copy(
        update={
            "databricks_admins_group": None,
            "databricks_datascientists_group": "",
            "databricks_analysts_group": "   ",
        }
    )
    workspace = _FakeWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)

    run_id = runner.submit(_request(dataset))

    # Falls back to the plain, pre-sharing submit signature -- proves the
    # feature is fully inert (not just empty-list) when unconfigured.
    call = workspace.jobs.submit_calls[0]
    assert call["access_control_list"] is None
    record = runner._find_job(run_id)
    assert record.status.value != "Failed"


def test_a_sharing_failure_falls_back_to_an_unshared_submit_rather_than_failing_the_run(settings, dataset):
    class _RefusingOnceJobs:
        def __init__(self):
            self.calls = []

        def submit(self, run_name=None, tasks=None, access_control_list=None):
            self.calls.append(access_control_list)
            if access_control_list is not None:
                raise RuntimeError("simulated: RESOURCE_DOES_NOT_EXIST, group ForecastIQ-Admins not found")
            return type("R", (), {"run_id": 99})()

    class _Workspace:
        def __init__(self):
            from test_databricks_runner import _FakeFiles

            self.files = _FakeFiles()
            self.jobs = _RefusingOnceJobs()

    workspace = _Workspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)

    run_id = runner.submit(_request(dataset))

    # First call attempted sharing and failed; the run still started via a
    # second, unshared call rather than surfacing as a failed run.
    assert len(workspace.jobs.calls) == 2
    assert workspace.jobs.calls[0] is not None
    assert workspace.jobs.calls[1] is None
    record = runner._find_job(run_id)
    assert record.databricks_run_id == 99
