"""GET /metadata/models is the one call the Configure step makes before
rendering model checkboxes and the horizon slider -- both answer the same
question ("what can this run actually be"), so both are asserted here
against the real route, not just the schema in isolation.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_principal
from app.auth.models import Principal, Role
from app.auth.rbac import permissions_for
from app.config.run_limits import DEFAULT_FORECAST_HORIZON, MAX_FORECAST_HORIZON, MIN_FORECAST_HORIZON
from app.main import app


def _principal(role: Role) -> Principal:
    return Principal(
        subject=f"user-{role.value}", display_name=role.value, roles=[role], permissions=permissions_for([role])
    )


@pytest.fixture
def client():
    app.dependency_overrides[get_current_principal] = lambda: _principal(Role.DATA_SCIENTIST)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_the_response_carries_the_shared_horizon_bounds(client):
    response = client.get("/metadata/models")

    assert response.status_code == 200
    horizon = response.json()["horizon"]
    assert horizon == {
        "min_months": MIN_FORECAST_HORIZON,
        "max_months": MAX_FORECAST_HORIZON,
        "default_months": DEFAULT_FORECAST_HORIZON,
    }


def test_every_candidate_model_is_reported(client):
    response = client.get("/metadata/models")

    ids = {m["id"] for m in response.json()["models"]}
    assert {"prophet", "arima", "lightgbm", "xgboost", "tft"} <= ids
