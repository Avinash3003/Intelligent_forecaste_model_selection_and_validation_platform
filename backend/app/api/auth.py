"""Authentication surface: what to sign in against, and who is signed in."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.dependencies import get_current_principal
from app.auth.models import Principal
from app.config.settings import Settings, get_settings
from app.schemas.auth import AuthConfigResponse, CurrentUserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/config", response_model=AuthConfigResponse, summary="Public Entra ID configuration for the frontend")
def get_auth_config(settings: Settings = Depends(get_settings)) -> AuthConfigResponse:
    """Public sign-in parameters.

    Intentionally unauthenticated — the frontend has to read this before
    it can obtain a token. It carries only public identifiers (see
    `AuthConfigResponse`), which is what lets the SPA hold no build-time
    Azure configuration of its own: one deployed frontend bundle can point
    at any tenant its backend names.
    """
    if not settings.auth_enabled:
        return AuthConfigResponse(auth_enabled=False)

    tenant = (settings.entra_tenant_id or "").strip() or None
    audience = (settings.entra_api_audience or "").strip() or None
    return AuthConfigResponse(
        auth_enabled=True,
        tenant_id=tenant,
        client_id=(settings.entra_spa_client_id or "").strip() or None,
        authority=f"{settings.entra_authority_host.rstrip('/')}/{tenant}" if tenant else None,
        # `.default` requests every app role/scope the SPA has been
        # consented for on the API, so adding a role in Azure needs no
        # frontend change.
        api_scope=f"{audience}/.default" if audience else None,
    )


@router.get("/me", response_model=CurrentUserResponse, summary="The signed-in user and their permissions")
def get_me(principal: Principal = Depends(get_current_principal)) -> CurrentUserResponse:
    return CurrentUserResponse(
        subject=principal.subject,
        display_name=principal.display_name,
        email=principal.email,
        roles=principal.roles,
        permissions=principal.permissions,
        is_development_identity=principal.is_development_identity,
    )
