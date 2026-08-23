"""Priority C — derived feature selection: backend registry, request
validation, and pass-through into the execution request.
"""

from __future__ import annotations

from pathlib import Path

from app.auth.dependencies import get_current_principal
from app.auth.models import Principal, Role
from app.auth.rbac import permissions_for
from app.main import app
from app.services.derived_feature_registry import (
    SUPPORTED_DERIVED_FEATURE_IDS,
    validate_derived_features,
)
from app.services.deployment_service import build_execution_request
from app.schemas.deployment import DeploymentRequest
from app.schemas.metadata import MetadataMapping
from fastapi.testclient import TestClient
import pytest


def test_registry_matches_the_engines_own_supported_feature_ids():
    # Deliberately hard-coded, not imported (the two live in separate
    # processes/dependency sets) — this is what keeps them from silently
    # drifting apart.
    assert SUPPORTED_DERIVED_FEATURE_IDS == {
        "lag_1", "lag_2", "lag_3", "lag_12",
        "rolling_mean_3", "rolling_mean_6",
        "month", "quarter",
    }


def test_validate_returns_only_the_unsupported_ids():
    assert validate_derived_features(["lag_1", "month"]) == []
    assert validate_derived_features(["lag_1", "not_real"]) == ["not_real"]
    assert validate_derived_features([]) == []


def _principal(role: Role) -> Principal:
    return Principal(subject=f"user-{role.value}", display_name=role.value, roles=[role], permissions=permissions_for([role]))


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_deploy_rejects_an_unsupported_derived_feature_name(client):
    app.dependency_overrides[get_current_principal] = lambda: _principal(Role.DATA_SCIENTIST)

    response = client.post(
        "/deploy",
        json={
            "file_id": "file-does-not-matter-validation-runs-first",
            "metadata": {"date_column": "date", "target_column": "sales", "key_columns": [], "feature_columns": []},
            "selected_models": ["xgboost"],
            "horizon": 12,
            "derived_features": ["lag_1", "not_a_real_feature"],
        },
    )

    assert response.status_code == 400
    assert "not_a_real_feature" in response.json()["detail"]


def test_deploy_accepts_a_fully_supported_selection_past_validation(client, monkeypatch):
    # Only proves validation lets a valid selection through to file
    # resolution (the next step) rather than rejecting it — not a full
    # end-to-end deploy, which belongs to the engine-level integration test.
    app.dependency_overrides[get_current_principal] = lambda: _principal(Role.DATA_SCIENTIST)

    response = client.post(
        "/deploy",
        json={
            "file_id": "file-does-not-exist",
            "metadata": {"date_column": "date", "target_column": "sales", "key_columns": [], "feature_columns": []},
            "selected_models": ["xgboost"],
            "horizon": 12,
            "derived_features": ["lag_1", "month"],
        },
    )

    # Past validation: fails on file resolution (404), not the 400 a bad
    # feature name would produce.
    assert response.status_code == 404


def test_build_execution_request_carries_the_selection_through_unchanged():
    request = DeploymentRequest(
        file_id="file-1",
        metadata=MetadataMapping(date_column="date", target_column="sales", key_columns=[], feature_columns=[]),
        selected_models=["xgboost"],
        horizon=12,
        derived_features=["lag_1", "month"],
    )
    principal = _principal(Role.DATA_SCIENTIST)

    execution_request = build_execution_request(request, Path("dataset.csv"), principal)

    assert execution_request.derived_features == ["lag_1", "month"]


def test_build_execution_request_preserves_an_explicit_empty_selection():
    request = DeploymentRequest(
        file_id="file-1",
        metadata=MetadataMapping(date_column="date", target_column="sales", key_columns=[], feature_columns=[]),
        selected_models=["xgboost"],
        horizon=12,
        derived_features=[],
    )
    principal = _principal(Role.DATA_SCIENTIST)

    execution_request = build_execution_request(request, Path("dataset.csv"), principal)

    assert execution_request.derived_features == []


def test_build_execution_request_default_is_none_not_an_empty_list():
    request = DeploymentRequest(
        file_id="file-1",
        metadata=MetadataMapping(date_column="date", target_column="sales", key_columns=[], feature_columns=[]),
        selected_models=["xgboost"],
        horizon=12,
    )
    principal = _principal(Role.DATA_SCIENTIST)

    execution_request = build_execution_request(request, Path("dataset.csv"), principal)

    assert execution_request.derived_features is None
