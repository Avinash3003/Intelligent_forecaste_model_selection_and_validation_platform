"""Every authenticated ForecastIQ user can read a run's MLflow record.

The product decision: a user must NOT need a separately provisioned
Databricks identity to see the MLflow run behind a forecast. That is served
by the backend reading the run through its own configured Databricks
credentials, exposed at /mlflow/runs/{run_id} and gated only by
MODEL_INSPECT — which every role holds.

These tests pin that so a future permission change cannot quietly turn the
in-app path into a role-restricted one and push users back toward needing a
Databricks account.
"""

from __future__ import annotations

import pytest

from app.auth.models import Permission, Role
from app.auth.rbac import ROLE_PERMISSIONS


@pytest.mark.parametrize("role", list(Role))
def test_every_role_can_inspect_the_mlflow_run_record(role):
    assert Permission.MODEL_INSPECT in ROLE_PERMISSIONS[role], (
        f"{role.value} lost MODEL_INSPECT — the in-app MLflow run view is the "
        "path that works without a Databricks account, so it must stay "
        "available to every authenticated role."
    )


def test_the_readonly_role_specifically_retains_it():
    """Analyst is the read-only role and the one most at risk of being
    trimmed; called out separately so the intent is unmissable."""
    assert Permission.MODEL_INSPECT in ROLE_PERMISSIONS[Role.ANALYST]


def test_inspecting_a_run_needs_no_write_shaped_permission():
    """Viewing an MLflow record must not require anything that could mutate
    a run — otherwise read-only users would be pushed to a higher role."""
    analyst = ROLE_PERMISSIONS[Role.ANALYST]
    for write_shaped in (
        Permission.FORECAST_RUN,
        Permission.RUN_CANCEL,
        Permission.DATASET_UPLOAD,
        Permission.ADMIN_MANAGE,
    ):
        assert write_shaped not in analyst
