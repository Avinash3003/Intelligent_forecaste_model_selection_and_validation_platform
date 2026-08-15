"""Who is making this request, and what they may do.

Everything authorizes against Permission, never a role name: roles are how
Entra describes a person, permissions are what the app gates. Keeping them
separate lets an operator remap a group without touching a route.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    """The three roles, matching the Entra app-role values exactly so a token's
    roles claim maps here with no translation table."""

    ADMIN = "Admin"
    DATA_SCIENTIST = "DataScientist"
    ANALYST = "Analyst"


class Permission(str, Enum):
    """One operation that can be granted or denied.

    Deliberately coarse: one per thing a user can do, not one per route.
    """

    DATASET_UPLOAD = "dataset:upload"
    DATASET_READ = "dataset:read"

    FORECAST_CONFIGURE = "forecast:configure"
    FORECAST_ESTIMATE = "forecast:estimate"
    FORECAST_RUN = "forecast:run"

    RUN_READ = "run:read"
    RUN_CANCEL = "run:cancel"

    RESULTS_READ = "results:read"
    # Model internals: MLflow runs, SHAP explainability, the debug view.
    MODEL_INSPECT = "model:inspect"

    ADMIN_MANAGE = "admin:manage"


class Principal(BaseModel):
    """The authenticated caller.

    roles is what Entra asserted; permissions is what we derived. Both are
    kept so "why was this allowed?" needs no re-derivation.
    """

    subject: str
    display_name: str | None = None
    email: str | None = None
    tenant_id: str | None = None
    roles: list[Role] = Field(default_factory=list)
    permissions: list[Permission] = Field(default_factory=list)
    # True when the local development bypass produced this principal rather
    # than a validated token. Surfaced to the frontend so a developer can
    # never mistake an unauthenticated local session for a real one.
    is_development_identity: bool = False

    def has(self, permission: Permission) -> bool:
        return permission in self.permissions

    @property
    def primary_role(self) -> Role | None:
        """Most privileged role held, for display only — authorization always
        goes through permissions, which unions every role."""
        for role in (Role.ADMIN, Role.DATA_SCIENTIST, Role.ANALYST):
            if role in self.roles:
                return role
        return None
