"""API-level cancellation and user-attribution security.

Covers what a unit test of the Runner/service layer alone cannot:
RBAC enforcement through the real dependency graph, and that a caller
cannot spoof `started_by` by stuffing it into a request body `DeploymentRequest`
has no field for.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_principal
from app.auth.models import Permission, Principal, Role
from app.auth.rbac import permissions_for
from app.main import app
from app.orchestration.exceptions import UnknownRunError
from app.orchestration.schemas import CancellationOutcome
from app.services.deployment_service import build_execution_request
from app.schemas.deployment import DeploymentRequest
from app.schemas.metadata import MetadataMapping


def _principal(role: Role, subject: str = "user-1", display_name: str = "Avinash Reddy") -> Principal:
    return Principal(subject=subject, display_name=display_name, roles=[role], permissions=permissions_for([role]))


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def as_role(role: Role, **kwargs) -> Principal:
    principal = _principal(role, **kwargs)
    app.dependency_overrides[get_current_principal] = lambda: principal
    return principal


class _FakeExecutor:
    def __init__(self, outcome=None, raises=None):
        self._outcome = outcome or CancellationOutcome(cancelled=True)
        self._raises = raises
        self.cancel_calls: list[tuple] = []

    def cancel(self, run_id, cancelled_by_user_id=None, cancelled_by_display_name=None):
        self.cancel_calls.append((run_id, cancelled_by_user_id, cancelled_by_display_name))
        if self._raises:
            raise self._raises
        return self._outcome


# ---------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------


def test_analyst_cannot_cancel_a_run(client):
    as_role(Role.ANALYST)
    assert client.post("/execution/whatever/cancel").status_code == 403


def test_data_scientist_can_cancel_a_run(client, monkeypatch):
    as_role(Role.DATA_SCIENTIST)
    fake = _FakeExecutor(CancellationOutcome(cancelled=True, cleanup_errors=[]))
    monkeypatch.setattr("app.api.execution.get_pipeline_executor", lambda: fake)

    response = client.post("/execution/fe-run-123/cancel")

    assert response.status_code == 200
    body = response.json()
    assert body == {"run_id": "fe-run-123", "cancelled": True, "cleanup_errors": []}


def test_admin_can_cancel_another_users_run(client, monkeypatch):
    """The scenario the spec calls out explicitly: an admin cancelling a
    run they did not start. RBAC here is role-based, not ownership-based —
    the same as every other run-read/inspect permission in this platform —
    so this is expected to succeed, not be refused."""
    as_role(Role.ADMIN, subject="admin-1", display_name="Admin User")
    fake = _FakeExecutor()
    monkeypatch.setattr("app.api.execution.get_pipeline_executor", lambda: fake)

    response = client.post("/execution/fe-run-someone-elses/cancel")

    assert response.status_code == 200
    assert fake.cancel_calls == [("fe-run-someone-elses", "admin-1", "Admin User")]


def test_cleanup_errors_are_reported_not_swallowed(client, monkeypatch):
    as_role(Role.ADMIN)
    fake = _FakeExecutor(CancellationOutcome(cancelled=True, cleanup_errors=["trained models: permission denied"]))
    monkeypatch.setattr("app.api.execution.get_pipeline_executor", lambda: fake)

    body = client.post("/execution/fe-run-1/cancel").json()

    assert body["cancelled"] is True
    assert body["cleanup_errors"] == ["trained models: permission denied"]


def test_unknown_run_id_is_a_404_not_a_silent_success(client, monkeypatch):
    as_role(Role.ADMIN)
    fake = _FakeExecutor(raises=UnknownRunError("no such run"))
    monkeypatch.setattr("app.api.execution.get_pipeline_executor", lambda: fake)

    assert client.post("/execution/does-not-exist/cancel").status_code == 404


def test_unauthenticated_caller_cannot_cancel(client):
    # No dependency override at all: the real Entra-token path, which in
    # this test environment (no bearer token supplied) refuses outright.
    from app.config.settings import get_settings

    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update={"auth_enabled": True})
    response = client.post("/execution/fe-run-1/cancel")
    assert response.status_code == 401


# ---------------------------------------------------------------------
# started_by cannot be spoofed by the caller
# ---------------------------------------------------------------------


def test_deployment_request_has_no_started_by_field_to_spoof():
    assert "started_by" not in DeploymentRequest.model_fields
    assert "started_by_user_id" not in DeploymentRequest.model_fields
    assert "started_by_display_name" not in DeploymentRequest.model_fields


def test_build_execution_request_always_derives_identity_from_the_principal(tmp_path):
    """Even if a client stuffs an unexpected `started_by`-shaped key into
    the JSON body, `DeploymentRequest` has no field to catch it in — extra
    keys are dropped by Pydantic — so the only identity that can ever reach
    `PipelineExecutionRequest` is the one this test passes as `principal`,
    which in the real route always comes from `Depends(require(...))`,
    never from request-body JSON.
    """
    request = DeploymentRequest.model_validate(
        {
            "file_id": "file-1",
            "metadata": {"date_column": "date", "target_column": "sales"},
            # An attacker's attempt — silently ignored, not an error, and
            # never reaches PipelineExecutionRequest.
            "started_by_display_name": "Not Actually The Caller",
            "started_by_user_id": "attacker-controlled",
        }
    )
    principal = _principal(Role.DATA_SCIENTIST, subject="real-user-id", display_name="Real Caller")

    execution_request = build_execution_request(request, tmp_path / "sales.csv", principal)

    assert execution_request.started_by_user_id == "real-user-id"
    assert execution_request.started_by_display_name == "Real Caller"
