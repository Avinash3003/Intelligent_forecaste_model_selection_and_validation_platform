"""Schemas for the authentication surface."""

from __future__ import annotations

from pydantic import BaseModel

from app.auth.models import Permission, Role


class AuthConfigResponse(BaseModel):
    """What the frontend needs to start an Entra sign-in.

    Every field here is a public identifier — a SPA client id, a tenant
    id and an authority URL are all visible in any browser that completes
    a sign-in. No secret is served from this endpoint, and it is the only
    unauthenticated route in the application: the frontend must be able to
    read it *before* it has a token.
    """

    auth_enabled: bool
    tenant_id: str | None = None
    client_id: str | None = None
    authority: str | None = None
    # The scope the SPA must request so the issued token targets this API
    # rather than Microsoft Graph.
    api_scope: str | None = None


class CurrentUserResponse(BaseModel):
    """The signed-in caller, as the application understands them.

    Returns `permissions` rather than only roles so the UI can hide
    actions it would be refused, without duplicating the role -> permission
    table client-side.
    """

    subject: str
    display_name: str | None = None
    email: str | None = None
    roles: list[Role]
    permissions: list[Permission]
    is_development_identity: bool = False
