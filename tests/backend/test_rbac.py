"""The role -> permission table is the whole authorization model, so it is
tested as a contract rather than incidentally through routes."""

from app.auth.models import Permission, Role
from app.auth.rbac import ROLE_PERMISSIONS, permissions_for


def test_analyst_can_view_but_never_execute_or_inspect():
    granted = ROLE_PERMISSIONS[Role.ANALYST]

    assert Permission.RESULTS_READ in granted
    assert Permission.RUN_READ in granted
    assert Permission.DATASET_READ in granted

    # The whole point of the Analyst role: read finished work, change nothing.
    for denied in (
        Permission.DATASET_UPLOAD,
        Permission.FORECAST_RUN,
        Permission.FORECAST_CONFIGURE,
        Permission.RUN_CANCEL,
        Permission.MODEL_INSPECT,
        Permission.ADMIN_MANAGE,
    ):
        assert denied not in granted, f"Analyst must not hold {denied}"


def test_data_scientist_can_run_but_not_administer():
    granted = ROLE_PERMISSIONS[Role.DATA_SCIENTIST]

    assert Permission.DATASET_UPLOAD in granted
    assert Permission.FORECAST_RUN in granted
    assert Permission.MODEL_INSPECT in granted
    assert Permission.ADMIN_MANAGE not in granted


def test_admin_holds_every_permission():
    assert set(ROLE_PERMISSIONS[Role.ADMIN]) == set(Permission)


def test_no_roles_grants_nothing():
    # An authenticated tenant user with no app-role assignment must not
    # fall through to viewer access.
    assert permissions_for([]) == []


def test_multiple_roles_union_rather_than_override():
    both = set(permissions_for([Role.ANALYST, Role.DATA_SCIENTIST]))
    assert both == set(ROLE_PERMISSIONS[Role.DATA_SCIENTIST])
    # Holding an extra, weaker role never removes anything.
    assert Permission.FORECAST_RUN in both


def test_permissions_are_returned_in_a_stable_order():
    assert permissions_for([Role.ADMIN]) == permissions_for([Role.ADMIN])
