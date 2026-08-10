"""Authentication (Entra ID) and authorization (RBAC) for the API layer."""

from app.auth.dependencies import get_current_principal, get_token_validator, require
from app.auth.models import Permission, Principal, Role
from app.auth.rbac import ROLE_PERMISSIONS, permissions_for

__all__ = [
    "ROLE_PERMISSIONS",
    "Permission",
    "Principal",
    "Role",
    "get_current_principal",
    "get_token_validator",
    "permissions_for",
    "require",
]
