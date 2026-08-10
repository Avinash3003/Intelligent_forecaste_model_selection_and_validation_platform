"""Identity and authorization vocabulary.

One `Principal` describes whoever is making the current request, whether
that identity came from a validated Entra ID access token or from the
local development bypass. Everything above this module authorizes against
`Permission`, never against a role name directly — roles are how Entra
describes a person, permissions are what the application actually gates,
and keeping them separate is what lets an operator re-map an Entra group
to a different role without touching a single route.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    """The three roles the platform distinguishes.

    Values match the `value` of the Entra ID **app roles** registered on
    the API application (see docs/PHASE_A_AZURE_SETUP.md), so a token's
    `roles` claim maps onto this enum with no translation table.
    """

    ADMIN = "Admin"
    DATA_SCIENTIST = "DataScientist"
    ANALYST = "Analyst"


class Permission(str, Enum):
    """A single application operation that can be granted or denied.

    Deliberately coarse — one entry per thing a user can actually *do* in
    the product, not one per HTTP route. Finer granularity would be
    invented structure: nothing in the product distinguishes, say,
    "read a run's status" from "read a run's stage trail".
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
    """The authenticated caller behind one request.

    `roles` is what the identity provider asserted; `permissions` is what
    this application derived from it. Both are carried so an audit answer
    ("why was this allowed?") does not require re-deriving the mapping.
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
        """The most privileged role held, for display purposes only.

        Never used for authorization — that always goes through
        `permissions`, which already unions every role the caller holds.
        """
        for role in (Role.ADMIN, Role.DATA_SCIENTIST, Role.ANALYST):
            if role in self.roles:
                return role
        return None
