"""Turns infrastructure failures into messages a user can act on.

Two jobs, deliberately separate:
  - friendly_message() translates the failures that have a real explanation
    (missing Azure OpenAI config, unreachable workspace, permissions).
  - redact() strips anything resembling a secret, token or internal URL.

Everything reaching a client goes through both, so an unrecognised error
degrades to a generic message rather than leaking a connection string.
"""

from __future__ import annotations

import re

GENERIC_MESSAGE = "Something went wrong on the platform. Please try again, or contact an administrator."

# Ordered most-specific first: the first pattern that matches wins, so a
# Databricks secret-resolution failure is explained as a configuration
# problem rather than as the generic "job could not start".
_TRANSLATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"secret.?resolution|X_SecretResolutionFailure|secret scope .* does not exist", re.I),
        "The forecast could not start because a required platform credential is not configured. "
        "An administrator needs to populate the ForecastIQ secret scope.",
    ),
    (
        re.compile(r"azure[_ ]?openai|openai.*(deployment|api key)", re.I),
        "Business insights are unavailable because the Azure OpenAI configuration is missing or invalid. "
        "The forecast itself is unaffected.",
    ),
    (
        re.compile(r"\b(401|unauthorized|invalid.?access.?token|token expired)\b", re.I),
        "The platform is not authorised to reach Azure Databricks. "
        "An administrator needs to check the service principal credentials.",
    ),
    (
        re.compile(r"\b(403|permission denied|forbidden|does not have permission)\b", re.I),
        "The platform does not have permission to perform this operation in Azure Databricks. "
        "An administrator needs to review its workspace access.",
    ),
    (
        re.compile(r"\b(quota|stockout|NotAvailableForSubscription|SkuNotAvailable|capacity)\b", re.I),
        "The forecast could not start because no compute capacity is currently available in Azure. "
        "Please retry later, or contact an administrator about the subscription quota.",
    ),
    (
        re.compile(r"\b(404|not found|RESOURCE_DOES_NOT_EXIST)\b.*job|job.*\b(404|not found|RESOURCE_DOES_NOT_EXIST)\b", re.I),
        "The forecasting job is not deployed in the connected Databricks workspace. "
        "An administrator needs to deploy the ForecastIQ bundle.",
    ),
    (
        re.compile(r"\b(timeout|timed out|connection (refused|reset|error)|unreachable|getaddrinfo)\b", re.I),
        "The platform could not reach Azure Databricks. Please try again shortly.",
    ),
    (
        re.compile(r"no such file|filenotfound|does not exist", re.I),
        "The requested dataset or run output could not be found. It may have been removed.",
    ),
)

# Anything matching these is never echoed back, whatever else is going on.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Databricks / Azure workspace and storage hosts.
    (re.compile(r"https?://[^\s'\"]+", re.I), "[internal endpoint]"),
    (re.compile(r"abfss://[^\s'\"]+", re.I), "[storage location]"),
    # Connection strings and key=value secrets.
    (re.compile(r"(AccountKey|SharedAccessSignature|password|client_secret|api[_-]?key)\s*=\s*[^\s;'\"]+", re.I), r"\1=[redacted]"),
    # Bearer tokens, Databricks PATs (dapi...), and JWTs.
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+", re.I), "Bearer [redacted]"),
    (re.compile(r"\bdapi[a-f0-9]{16,}\b", re.I), "[redacted]"),
    (re.compile(r"\beyJ[A-Za-z0-9._\-]{20,}", re.I), "[redacted]"),
    # Absolute filesystem paths, which disclose server layout.
    (re.compile(r"(?<![\w/])/(?:home|root|mnt|opt|usr|var|etc)/[^\s'\"]*"), "[path]"),
)


def redact(text: str) -> str:
    """Strip secrets, tokens, endpoints and server paths from `text`."""
    cleaned = text or ""
    for pattern, replacement in _REDACTIONS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned.strip()


def friendly_message(error: BaseException | str, *, fallback: str = GENERIC_MESSAGE) -> str:
    """A message safe to show a user.

    Recognised failures are explained in product terms; anything else falls
    back to the generic text, since we cannot know what it contains.
    """
    raw = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    for pattern, message in _TRANSLATIONS:
        if pattern.search(raw):
            return message
    return fallback


def safe_detail(error: BaseException | str, *, fallback: str = GENERIC_MESSAGE) -> str:
    """friendly_message then redact — what routes should call.

    Redaction runs even on translated text, so a future translation that
    interpolates the original cannot reintroduce a leak.
    """
    return redact(friendly_message(error, fallback=fallback)) or fallback
