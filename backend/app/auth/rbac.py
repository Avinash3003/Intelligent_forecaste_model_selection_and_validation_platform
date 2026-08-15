"""Role -> permission mapping: the whole authorization model.

Several roles union rather than "highest wins", so adding a role can only
widen access, never silently narrow it.
"""

from __future__ import annotations

from app.auth.models import Permission, Role

# Read-only access to finished work. The floor every role stands on.
_VIEWER_PERMISSIONS: frozenset[Permission] = frozenset(
    {
        Permission.DATASET_READ,
        Permission.RUN_READ,
        Permission.RESULTS_READ,
    }
)

# Everything needed to take a dataset from upload to a finished forecast.
_PRACTITIONER_PERMISSIONS: frozenset[Permission] = _VIEWER_PERMISSIONS | frozenset(
    {
        Permission.DATASET_UPLOAD,
        Permission.FORECAST_CONFIGURE,
        Permission.FORECAST_ESTIMATE,
        Permission.FORECAST_RUN,
        Permission.RUN_CANCEL,
        Permission.MODEL_INSPECT,
    }
)

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    # Everything a data scientist can do, plus platform configuration.
    Role.ADMIN: _PRACTITIONER_PERMISSIONS | frozenset({Permission.ADMIN_MANAGE}),
    Role.DATA_SCIENTIST: _PRACTITIONER_PERMISSIONS,
    # Deliberately cannot upload, run, cancel, or inspect model internals.
    Role.ANALYST: _VIEWER_PERMISSIONS,
}


def permissions_for(roles: list[Role]) -> list[Permission]:
    """Every permission the given roles grant, in a stable order.

    No roles means no permissions: an authenticated user with no app role is
    authenticated but not authorized.
    """
    granted: set[Permission] = set()
    for role in roles:
        granted |= ROLE_PERMISSIONS.get(role, frozenset())
    # Enum declaration order, so responses and tests are deterministic.
    return [permission for permission in Permission if permission in granted]
