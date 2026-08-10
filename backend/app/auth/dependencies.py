"""FastAPI dependencies that put authentication and RBAC in front of routes.

A route declares what it needs — `Depends(require(Permission.FORECAST_RUN))`
— and never inspects a token, a role, or a header itself. That is what
keeps the authorization model in one readable table (`app/auth/rbac.py`)
instead of scattered across route bodies.
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

    With `AUTH_ENABLED=false` this returns the local development identity
    without inspecting any header at all — see
    `app.auth.entra.development_principal` for why that cannot leak into
    a real deployment.
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
    """Dependency factory: allow the request only if the caller holds
    *every* listed permission.

    Returns the `Principal` so a route that needs the caller's identity
    (to record who submitted a run, say) gets it from the same dependency
    that authorized the call, with no second lookup that could disagree.
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
