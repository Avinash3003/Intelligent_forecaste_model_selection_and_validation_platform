"""End-to-end authorization through the real FastAPI app.

Uses the running application with its dependency graph intact, so these
cover what actually protects the API — not the RBAC table in isolation.
"""

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_principal
from app.auth.models import Permission, Principal, Role
from app.auth.rbac import permissions_for
from app.main import app


def _principal(role: Role) -> Principal:
    return Principal(
        subject=f"user-{role.value}",
        display_name=role.value,
        roles=[role],
        permissions=permissions_for([role]),
    )


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def as_role(role: Role) -> None:
    app.dependency_overrides[get_current_principal] = lambda: _principal(role)


def as_unassigned() -> None:
    """A real tenant user with no app role — authenticated, not authorized."""
    app.dependency_overrides[get_current_principal] = lambda: Principal(
        subject="user-none", roles=[], permissions=[]
    )


def test_health_needs_no_authentication(client):
    assert client.get("/health").status_code == 200


def test_auth_config_is_readable_before_sign_in(client):
    # The frontend cannot obtain a token without first reading this.
    response = client.get("/auth/config")
    assert response.status_code == 200
    body = response.json()
    assert "auth_enabled" in body
    # No secret may ever appear here.
    assert "client_secret" not in body
    assert "api_key" not in body


def test_me_reports_roles_and_permissions(client):
    as_role(Role.DATA_SCIENTIST)
    body = client.get("/auth/me").json()

    assert body["roles"] == ["DataScientist"]
    assert Permission.FORECAST_RUN.value in body["permissions"]
    assert Permission.ADMIN_MANAGE.value not in body["permissions"]


def test_analyst_is_refused_upload_and_run(client):
    as_role(Role.ANALYST)

    assert client.post("/upload", files={"file": ("x.csv", b"a,b\n1,2\n")}).status_code == 403
    assert client.post("/deploy", json={
        "file_id": "irrelevant",
        "metadata": {"date_column": "d", "target_column": "t"},
    }).status_code == 403
    assert client.post("/execution/submit", json={
        "file_id": "irrelevant",
        "metadata": {"date_column": "d", "target_column": "t"},
    }).status_code == 403


def test_analyst_may_read_run_history(client):
    as_role(Role.ANALYST)
    assert client.get("/deployments").status_code == 200


def test_analyst_may_inspect_model_internals(client):
    # Experiments (mlflow_view) and Observability (llmops/debug) are
    # read-only views of finished runs, same as Results — the Analyst role
    # can see them too. Past the permission gate, so an unknown run is a
    # 404 rather than a 403.
    as_role(Role.ANALYST)
    assert client.get("/mlflow/runs/whatever").status_code == 404
    assert client.get("/results/whatever/debug").status_code == 404


def test_data_scientist_may_inspect_model_internals(client):
    as_role(Role.DATA_SCIENTIST)
    # Past the permission gate, so an unknown run is a 404 rather than 403.
    assert client.get("/mlflow/runs/whatever").status_code == 404


def test_user_with_no_role_is_refused_everything(client):
    as_unassigned()
    assert client.get("/deployments").status_code == 403
    assert client.get("/results/whatever").status_code == 403
    assert client.post("/upload", files={"file": ("x.csv", b"a,b\n1,2\n")}).status_code == 403
    # Still allowed to learn who they are, which is what lets the UI tell
    # them they need a role assignment.
    assert client.get("/auth/me").status_code == 200


def test_validation_errors_do_not_leak_internal_structure(client):
    as_role(Role.DATA_SCIENTIST)
    response = client.post("/metadata/validate", json={"file_id": "x"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, str)
    assert "pydantic" not in detail.lower()
    assert "loc" not in detail
