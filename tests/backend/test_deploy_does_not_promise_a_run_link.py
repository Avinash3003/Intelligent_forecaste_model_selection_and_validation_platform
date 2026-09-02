"""The Databricks run link is not knowable at submission, and must not be
claimed to be.

`DatabricksRunner.submit` stages the dataset and triggers the job on a
background thread, so `/deploy` can answer in milliseconds — a real 17.3 MB
dataset takes nearly a minute to upload on its own. That means when /deploy
returns there is no Databricks run id yet, and therefore no run page URL.

An earlier attempt returned the URL in the submission response. It was
always null in practice, which is worse than absent: a field that exists
but is never populated reads as "this run has no link" rather than "ask
again in a moment". The UI polls `/deployments/{run_id}` instead, where the
link appears a poll or two later — while the run is still starting, which
is when someone wants to open it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.auth.models import Principal, Role
from app.orchestration.schemas import JobStatus
from app.schemas.deployment import DeploymentRequest, DeploymentResponse, MetadataMapping
from app.services.deployment_service import DeploymentService


class _Executor:
    def __init__(self):
        self.get_run_calls: list[str] = []

    def execute(self, request):
        return "dbx-run-abc123"

    def get_status(self, run_id):
        return JobStatus.PENDING

    def get_run(self, run_id):
        self.get_run_calls.append(run_id)
        return SimpleNamespace(databricks_run_url=None)


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


def test_the_submission_response_does_not_carry_a_run_link():
    """No field promising something that is never there at this point."""
    assert "databricks_run_url" not in DeploymentResponse.model_fields


def test_submitting_does_not_wait_on_a_link_lookup():
    """The link costs a workspace round trip and cannot succeed yet, so the
    submission path must not spend one asking."""
    executor = _Executor()

    response = DeploymentService(executor=executor, upload_service=_Uploads()).deploy(_request(), _principal())

    assert response.run_id == "dbx-run-abc123"
    assert executor.get_run_calls == []
