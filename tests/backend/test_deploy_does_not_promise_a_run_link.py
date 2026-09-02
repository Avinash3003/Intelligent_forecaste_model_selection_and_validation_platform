"""The Databricks run link ships with the submission, not with the result.

Watching a run execute is only possible while it is executing, so a link
that appears once the run reaches history arrives exactly too late. The
submission confirmation used to say "no need to open Databricks" and offer
nothing; the run id the link is built from is known the moment the job is
triggered, so the URL travels back with the run id.

A missing link is never an error: a run that submitted fine is not failed
by the absence of a convenience.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.auth.models import Principal, Role
from app.orchestration.schemas import JobStatus
from app.schemas.deployment import DeploymentRequest, MetadataMapping
from app.services.deployment_service import DeploymentService

RUN_URL = "https://adb-1.4.azuredatabricks.net/?o=1#job/7/run/42"


class _Executor:
    """Stands in for the Pipeline Executor: submits, then reports."""

    def __init__(self, listing=SimpleNamespace(databricks_run_url=RUN_URL)):
        self._listing = listing
        self.get_run_calls: list[str] = []

    def execute(self, request):
        return "dbx-run-abc123"

    def get_status(self, run_id):
        return JobStatus.PENDING

    def get_run(self, run_id):
        self.get_run_calls.append(run_id)
        return self._listing


class _Uploads:
    def resolve(self, file_id):
        return Path("/tmp/sales.csv"), "sales.csv"


def _request():
    return DeploymentRequest(
        file_id="file-1",
        dataset_name="sales.csv",
        metadata=MetadataMapping(
            date_column="date", target_column="sales", key_columns=["store"], feature_columns=[]
        ),
        selected_models=["arima"],
        horizon=12,
    )


def _principal():
    return Principal(subject="u1", display_name="Test User", email="u1@example.com", role=Role.DATA_SCIENTIST)


@pytest.fixture(autouse=True)
def _no_compute_refusal(monkeypatch):
    """The compute gates are their own tests; this file is about the link."""
    import app.services.deployment_service as module

    monkeypatch.setattr(module, "_reject_unsupported_models", lambda request, settings: None)


def test_the_submission_response_carries_the_live_run_link():
    executor = _Executor()

    response = DeploymentService(executor=executor, upload_service=_Uploads()).deploy(_request(), _principal())

    assert response.run_id == "dbx-run-abc123"
    assert response.databricks_run_url == RUN_URL
    assert executor.get_run_calls == ["dbx-run-abc123"]


def test_a_run_with_no_link_still_submits():
    """Local execution, and any run Databricks gave no URL for."""
    executor = _Executor(listing=SimpleNamespace(databricks_run_url=None))

    response = DeploymentService(executor=executor, upload_service=_Uploads()).deploy(_request(), _principal())

    assert response.run_id == "dbx-run-abc123"
    assert response.databricks_run_url is None


def test_a_failure_looking_up_the_link_never_fails_the_submission():
    class _Exploding(_Executor):
        def get_run(self, run_id):
            raise RuntimeError("workspace unreachable")

    response = DeploymentService(executor=_Exploding(), upload_service=_Uploads()).deploy(_request(), _principal())

    assert response.run_id == "dbx-run-abc123"
    assert response.databricks_run_url is None
