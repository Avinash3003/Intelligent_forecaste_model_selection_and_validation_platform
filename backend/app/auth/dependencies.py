"""Puts authentication and RBAC in front of routes.

A route declares what it needs and never inspects a token or header itself,
which keeps the whole authorization model in one table (rbac.py).
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.entra import EntraTokenValidator, TokenValidationError, development_principal
from app.auth.models import Permission, Principal
from app.config.settings import Settings, get_settings

# auto_error=False so a missing header produces this module's own 401 with
# a readable message, rather than FastAPI's bare "Not authenticated".
_bearer = HTTPBearer(auto_error=False, scheme_name="EntraID")


@lru_cache
def get_token_validator() -> EntraTokenValidator:
    """Process-wide validator, so the JWKS/discovery cache is shared."""
    return EntraTokenValidator(get_settings())


def get_current_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> Principal:
    """The authenticated caller, or 401.

    With AUTH_ENABLED=false it returns the development identity without
    reading any header.
    """
    if not settings.auth_enabled:
        principal = development_principal(settings)
        request.state.principal = principal
        return principal

    if credentials is None or not (credentials.credentials or "").strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please sign in to continue.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        principal = get_token_validator().validate(credentials.credentials)
    except TokenValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    request.state.principal = principal
    return principal


def require(*permissions: Permission):
    """Allow the request only if the caller holds every listed permission.

    Returns the Principal, so a route needing the caller's identity gets it
    from the same dependency that authorized the call.
    """

    def _dependency(principal: Principal = Depends(get_current_principal)) -> Principal:
        missing = [permission for permission in permissions if not principal.has(permission)]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                # Names the capability, never the internal permission
                # string or the caller's role list.
                detail="Your role does not allow this action. Contact a ForecastIQ administrator for access.",
            )
        return principal

    return _dependency
