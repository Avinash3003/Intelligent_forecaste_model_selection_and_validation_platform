"""Entra ID access-token validation, exercised against real signed JWTs.

Tokens are minted here with a controlled RSA key and the validator's key
lookup is pointed at that key, so these assert the *validation* logic —
signature, audience, issuer, expiry, and the roles -> permissions mapping —
without needing a live interactive sign-in.

The tenant values are the real ones this deployment is configured against
(see backend/.env), so an audience or issuer that would be rejected in
production is rejected here too.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from app.auth.entra import EntraTokenValidator, TokenValidationError
from app.auth.models import Permission, Role
from app.config.settings import Settings

TENANT = "7388a08a-fd82-4a91-973d-ed37e9ca568a"
API_CLIENT_ID = "2c51f53a-fb90-4819-bc63-04093a045b32"
API_AUDIENCE = f"api://{API_CLIENT_ID}"
ISSUER = f"https://login.microsoftonline.com/{TENANT}/v2.0"


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def settings():
    return Settings(
        auth_enabled=True,
        entra_tenant_id=TENANT,
        entra_api_client_id=API_CLIENT_ID,
        entra_api_audience=API_AUDIENCE,
    )


@pytest.fixture
def validator(settings, signing_key, monkeypatch):
    """A validator whose key lookup returns our test key.

    Only the network-bound discovery step is replaced; every assertion
    below still runs through the real `validate()` code path.
    """
    instance = EntraTokenValidator(settings)
    fake_jwks = SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key=signing_key.public_key())
    )
    monkeypatch.setattr(
        instance,
        "_ensure_discovery",
        lambda: (fake_jwks, [ISSUER, f"https://login.microsoftonline.com/{TENANT}/"]),
    )
    return instance


def mint(signing_key, **overrides) -> str:
    now = int(time.time())
    claims = {
        "aud": API_AUDIENCE,
        "iss": ISSUER,
        "iat": now,
        "nbf": now,
        "exp": now + 3600,
        "tid": TENANT,
        "oid": "ba40c78a-9e2e-4bd3-85e2-3a0cc9bd3f09",
        "sub": "subject-abc",
        "name": "Avinash A",
        "preferred_username": "ambatiniharika.apply_gmail.com#EXT#@ambatiniharikaapplygmail.onmicrosoft.com",
        "roles": ["Admin"],
    }
    claims.update(overrides)
    claims = {key: value for key, value in claims.items() if value is not None}
    return jwt.encode(claims, signing_key, algorithm="RS256")


# ---------------------------------------------------------------------
# Accepted tokens
# ---------------------------------------------------------------------


def test_valid_admin_token_yields_an_admin_principal(validator, signing_key):
    principal = validator.validate(mint(signing_key))

    assert principal.roles == [Role.ADMIN]
    assert principal.display_name == "Avinash A"
    # `oid` is preferred over `sub` — stable per tenant, unlike `sub`.
    assert principal.subject == "ba40c78a-9e2e-4bd3-85e2-3a0cc9bd3f09"
    assert principal.tenant_id == TENANT
    assert principal.is_development_identity is False
    assert set(principal.permissions) == set(Permission)


def test_v2_tokens_carrying_the_client_id_as_audience_are_accepted(validator, signing_key):
    """The audience this deployment actually issues.

    With `requestedAccessTokenVersion: 2` set on the API registration,
    Entra puts the API's *client id* in `aud`, not the App ID URI. A
    validator that only accepted the URI would reject every real token.
    """
    principal = validator.validate(mint(signing_key, aud=API_CLIENT_ID))
    assert principal.roles == [Role.ADMIN]


def test_v1_issuer_form_is_also_accepted(validator, signing_key):
    token = mint(signing_key, iss=f"https://login.microsoftonline.com/{TENANT}/")
    assert validator.validate(token).roles == [Role.ADMIN]


@pytest.mark.parametrize(
    "claim_roles,expected",
    [
        (["Admin"], [Role.ADMIN]),
        (["DataScientist"], [Role.DATA_SCIENTIST]),
        (["Analyst"], [Role.ANALYST]),
        (["Analyst", "DataScientist"], [Role.DATA_SCIENTIST, Role.ANALYST]),
    ],
)
def test_every_configured_app_role_maps(validator, signing_key, claim_roles, expected):
    principal = validator.validate(mint(signing_key, roles=claim_roles))
    assert sorted(r.value for r in principal.roles) == sorted(r.value for r in expected)


def test_role_matching_is_case_insensitive(validator, signing_key):
    assert validator.validate(mint(signing_key, roles=["datascientist"])).roles == [Role.DATA_SCIENTIST]


# ---------------------------------------------------------------------
# Rejected tokens
# ---------------------------------------------------------------------


def test_token_for_another_audience_is_rejected(validator, signing_key):
    # A Microsoft Graph token is a valid Entra token — it is simply not
    # ours, and accepting it would let any tenant app call this API.
    token = mint(signing_key, aud="00000003-0000-0000-c000-000000000000")
    with pytest.raises(TokenValidationError, match="not valid for the ForecastIQ API"):
        validator.validate(token)


def test_token_from_another_tenant_is_rejected(validator, signing_key):
    other = "11111111-2222-3333-4444-555555555555"
    token = mint(signing_key, iss=f"https://login.microsoftonline.com/{other}/v2.0")
    with pytest.raises(TokenValidationError):
        validator.validate(token)


def test_expired_token_is_rejected(validator, signing_key):
    now = int(time.time())
    token = mint(signing_key, exp=now - 60, iat=now - 3600, nbf=now - 3600)
    with pytest.raises(TokenValidationError, match="expired"):
        validator.validate(token)


def test_token_signed_by_a_different_key_is_rejected(validator):
    attacker = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(TokenValidationError):
        validator.validate(mint(attacker))


def test_tampered_token_is_rejected(validator, signing_key):
    token = mint(signing_key)
    header, payload, signature = token.split(".")
    with pytest.raises(TokenValidationError):
        validator.validate(f"{header}.{payload}x.{signature}")


def test_unsigned_token_is_rejected(validator, signing_key):
    # The classic alg=none downgrade. `algorithms=["RS256"]` must refuse it.
    now = int(time.time())
    token = jwt.encode(
        {"aud": API_AUDIENCE, "iss": ISSUER, "exp": now + 3600, "roles": ["Admin"]},
        key="",
        algorithm="none",
    )
    with pytest.raises(TokenValidationError):
        validator.validate(token)


def test_garbage_is_rejected(validator):
    with pytest.raises(TokenValidationError):
        validator.validate("not-a-jwt")


# ---------------------------------------------------------------------
# Authorization consequences
# ---------------------------------------------------------------------


def test_token_with_no_roles_claim_is_authenticated_but_powerless(validator, signing_key):
    principal = validator.validate(mint(signing_key, roles=None))

    assert principal.roles == []
    assert principal.permissions == []
    # The property that matters: a tenant member with no app-role
    # assignment must not inherit read access to results.
    assert not principal.has(Permission.RESULTS_READ)


def test_unrecognised_role_values_are_dropped_not_guessed(validator, signing_key):
    principal = validator.validate(mint(signing_key, roles=["Administrator", "SuperUser"]))
    assert principal.roles == []


def test_group_claim_maps_to_a_role_when_no_app_roles_are_present(signing_key, monkeypatch):
    group_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    settings = Settings(
        auth_enabled=True,
        entra_tenant_id=TENANT,
        entra_api_client_id=API_CLIENT_ID,
        entra_api_audience=API_AUDIENCE,
        entra_group_role_map=f'{{"{group_id}": "Analyst"}}',
    )
    instance = EntraTokenValidator(settings)
    fake_jwks = SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key=signing_key.public_key())
    )
    monkeypatch.setattr(instance, "_ensure_discovery", lambda: (fake_jwks, [ISSUER]))

    principal = instance.validate(mint(signing_key, roles=None, groups=[group_id]))
    assert principal.roles == [Role.ANALYST]

    # App roles win when both are present.
    principal = instance.validate(mint(signing_key, roles=["Admin"], groups=[group_id]))
    assert principal.roles == [Role.ADMIN]


# ---------------------------------------------------------------------
# Through the real HTTP stack
# ---------------------------------------------------------------------


def test_bearer_token_authenticates_a_real_api_request(validator, signing_key, monkeypatch):
    """The full path: Authorization header -> validation -> RBAC -> route."""
    from app.auth import dependencies
    from app.main import app

    monkeypatch.setattr(dependencies, "get_token_validator", lambda: validator)
    monkeypatch.setattr(dependencies.get_settings(), "auth_enabled", True, raising=False)

    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {mint(signing_key)}"}
        me = client.get("/auth/me", headers=headers)
        assert me.status_code == 200
        assert me.json()["roles"] == ["Admin"]
        assert Permission.ADMIN_MANAGE.value in me.json()["permissions"]

        # An Admin-only-through-RBAC surface answers normally.
        assert client.get("/deployments", headers=headers).status_code == 200

        # No token at all.
        assert client.get("/auth/me").status_code == 401
        # A token this API must not accept.
        bad = {"Authorization": f"Bearer {mint(signing_key, aud='some-other-api')}"}
        assert client.get("/deployments", headers=bad).status_code == 401
