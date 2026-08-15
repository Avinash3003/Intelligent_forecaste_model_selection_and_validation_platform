"""Validates the Entra ID access tokens the frontend sends.

The only place a bearer token is trusted or rejected. Checks, in order:
the RS256 signature against the tenant's published keys, aud (issued for
this API, not Graph), iss (the configured tenant), and exp/nbf.

No client secret is involved — validation needs only public keys — so this
service holds no Entra credential at all.
"""

from __future__ import annotations

import threading
import time

import jwt
import requests
from jwt import PyJWKClient

from app.auth.models import Permission, Principal, Role
from app.auth.rbac import permissions_for
from app.config.settings import Settings


class AuthConfigurationError(RuntimeError):
    """Authentication is switched on but not configured well enough to run."""


class TokenValidationError(Exception):
    """Token missing, malformed, expired or not ours. The message is safe to
    return: it never embeds the token, the key, or library internals."""


class EntraTokenValidator:
    """Validates tokens and maps them to a Principal.

    One instance per process. Keys and the discovery document are cached, so
    a steady-state request does no network I/O.
    """

    _DISCOVERY_TTL_SECONDS = 3600.0

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._jwks_client: PyJWKClient | None = None
        self._issuers: list[str] = []
        self._discovered_at: float = 0.0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def require_configuration(self) -> None:
        """Fail at startup, not as a 401 on every request, when auth is on but
        no tenant/audience is configured."""
        missing = [
            name
            for name, value in (
                ("ENTRA_TENANT_ID", self._settings.entra_tenant_id),
                ("ENTRA_API_AUDIENCE", self._settings.entra_api_audience),
            )
            if not (value or "").strip()
        ]
        if missing:
            raise AuthConfigurationError(
                "Authentication is enabled but not configured. Set: " + ", ".join(missing) + "."
            )

    @property
    def _discovery_url(self) -> str:
        return (
            f"{self._settings.entra_authority_host.rstrip('/')}/"
            f"{self._settings.entra_tenant_id}/v2.0/.well-known/openid-configuration"
        )

    def _ensure_discovery(self) -> tuple[PyJWKClient, list[str]]:
        with self._lock:
            fresh = time.monotonic() - self._discovered_at < self._DISCOVERY_TTL_SECONDS
            if self._jwks_client is not None and fresh:
                return self._jwks_client, self._issuers

            try:
                response = requests.get(self._discovery_url, timeout=10)
                response.raise_for_status()
                document = response.json()
            except requests.RequestException as exc:
                # Never leak the discovery URL or the transport error: both
                # name the tenant.
                raise TokenValidationError(
                    "Sign-in could not be verified because the identity provider is unreachable. "
                    "Please try again shortly."
                ) from exc

            jwks_uri = document.get("jwks_uri")
            issuer = document.get("issuer")
            if not jwks_uri or not issuer:
                raise TokenValidationError("Sign-in could not be verified. Please try again shortly.")

            self._jwks_client = PyJWKClient(jwks_uri, cache_keys=True)
            # Entra issues v2.0 tokens with the `/v2.0` issuer and v1.0
            # tokens without it. Which one arrives depends on the app
            # registration's `accessTokenAcceptedVersion`, which is an
            # Azure-side setting this API does not control — so accept
            # both forms of *this tenant's* issuer rather than rejecting
            # valid tokens over a trailing path segment.
            self._issuers = [issuer, issuer.replace("/v2.0", "").rstrip("/") + "/"]
            self._discovered_at = time.monotonic()
            return self._jwks_client, self._issuers

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, token: str) -> Principal:
        """The caller a token identifies. Raises TokenValidationError if not
        acceptable, with a message safe to show a user."""
        self.require_configuration()
        jwks_client, issuers = self._ensure_discovery()

        try:
            signing_key = jwks_client.get_signing_key_from_jwt(token)
        except Exception as exc:  # PyJWKClient raises several unrelated types
            raise TokenValidationError("Your session could not be verified. Please sign in again.") from exc

        audiences = [self._settings.entra_api_audience]
        # A token requested with the `api://<client-id>/<scope>` scope
        # carries either the App ID URI or the bare client id as `aud`,
        # depending on the app registration. Accepting the client id too
        # avoids a class of "correctly configured but still rejected".
        if self._settings.entra_api_client_id:
            audiences.append(self._settings.entra_api_client_id)

        last_error: Exception | None = None
        for issuer in issuers:
            try:
                claims = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=["RS256"],
                    audience=audiences,
                    issuer=issuer,
                    options={"require": ["exp", "aud", "iss"]},
                )
                return self._to_principal(claims)
            except jwt.InvalidIssuerError as exc:
                last_error = exc
                continue
            except jwt.ExpiredSignatureError as exc:
                raise TokenValidationError("Your session has expired. Please sign in again.") from exc
            except jwt.InvalidAudienceError as exc:
                raise TokenValidationError(
                    "This sign-in is not valid for the ForecastIQ API. Please sign in again."
                ) from exc
            except jwt.PyJWTError as exc:
                raise TokenValidationError("Your session could not be verified. Please sign in again.") from exc

        raise TokenValidationError("Your session could not be verified. Please sign in again.") from last_error

    def _to_principal(self, claims: dict) -> Principal:
        roles = self._roles_from_claims(claims)
        return Principal(
            # `oid` is the stable per-tenant object id; `sub` is
            # per-application and rotates if the app registration changes,
            # so `oid` is the better identity to record against a run.
            subject=str(claims.get("oid") or claims.get("sub") or "unknown"),
            display_name=claims.get("name"),
            email=claims.get("preferred_username") or claims.get("upn") or claims.get("email"),
            tenant_id=claims.get("tid"),
            roles=roles,
            permissions=permissions_for(roles),
        )

    def _roles_from_claims(self, claims: dict) -> list[Role]:
        """App roles first, then group mapping.

        App roles are the supported mechanism; group ids are a fallback for
        tenants that manage access through existing security groups.
        """
        found: list[Role] = []

        for value in claims.get("roles") or []:
            role = _parse_role(value)
            if role is not None and role not in found:
                found.append(role)

        if not found:
            mapping = self._settings.entra_group_role_map_parsed
            for group_id in claims.get("groups") or []:
                role = _parse_role(mapping.get(str(group_id), ""))
                if role is not None and role not in found:
                    found.append(role)

        return found


def _parse_role(value: str) -> Role | None:
    """Match a claim to a Role, case-insensitively.

    An unrecognised value is dropped, never guessed, so a typo in an app-role
    definition denies access rather than granting the nearest match.
    """
    candidate = (value or "").strip().lower()
    for role in Role:
        if role.value.lower() == candidate:
            return role
    return None


def development_principal(settings: Settings) -> Principal:
    """The identity used when AUTH_ENABLED=false, so the app runs with no tenant.

    Reachable only through that flag, marked as a development identity in
    every response, and main.py refuses to start if it is combined with a
    production APP_ENV — so it cannot become an accidental auth bypass.
    """
    role = _parse_role(settings.dev_identity_role) or Role.ADMIN
    return Principal(
        subject="local-development",
        display_name="Local Developer",
        email=None,
        tenant_id=None,
        roles=[role],
        permissions=permissions_for([role]),
        is_development_identity=True,
    )


__all__ = [
    "AuthConfigurationError",
    "EntraTokenValidator",
    "Permission",
    "Principal",
    "Role",
    "TokenValidationError",
    "development_principal",
]
