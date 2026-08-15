"""Schemas for the authentication surface."""

from __future__ import annotations

from pydantic import BaseModel

from app.auth.models import Permission, Role


class AuthConfigResponse(BaseModel):
    """What the frontend needs to start a sign-in.

    Public identifiers only — all visible in any browser that signs in. No
    secret is served here, and this is the app's only unauthenticated route.
    """

    auth_enabled: bool
    tenant_id: str | None = None
    client_id: str | None = None
    authority: str | None = None
    # The scope the SPA must request so the issued token targets this API
    # rather than Microsoft Graph.
    api_scope: str | None = None


class CurrentUserResponse(BaseModel):
    """The signed-in caller.

    Carries permissions, not just roles, so the UI can hide actions it would
    be refused without duplicating the mapping table.
    """

    subject: str
    display_name: str | None = None
    email: str | None = None
    roles: list[Role]
    permissions: list[Permission]
    is_development_identity: bool = False
