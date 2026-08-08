"""LLM Business Insights configuration (Section 6.12).

The platform uses Azure OpenAI exclusively — there is no provider
abstraction to configure a choice between. Every value the Azure OpenAI
Chat Completions call needs (endpoint, credential, deployment, API
version) lives here, centralized, and is populated from environment
variables rather than ever being hardcoded. A prompt's *wording* is a
separate concern, kept in `forecast_engine/llm/prompts/`.

Credentials are read from the environment once, at config-construction
time, and held only in memory for the lifetime of the run — never written
to a log, a report, or `PipelineResult`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Standard Azure OpenAI environment variable names. Documented as constants
# (rather than buried in a docstring) so `LLMConfig.from_env()` and any
# deployment's `.env` file stay in sync by construction.
AZURE_OPENAI_ENDPOINT_ENV_VAR = "AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_API_KEY_ENV_VAR = "AZURE_OPENAI_API_KEY"
AZURE_OPENAI_DEPLOYMENT_NAME_ENV_VAR = "AZURE_OPENAI_DEPLOYMENT_NAME"
AZURE_OPENAI_API_VERSION_ENV_VAR = "AZURE_OPENAI_API_VERSION"

# Azure OpenAI requires an explicit API version; this is the fallback used
# only when the environment variable above is not set.
_DEFAULT_API_VERSION = "2024-10-21"


@dataclass(frozen=True)
class LLMConfig:
    """Root configuration for the LLM Insight Engine — Azure OpenAI only."""

    enabled: bool = True

    # Resource endpoint, e.g. "https://<resource>.openai.azure.com/".
    endpoint: str | None = None

    # Secret credential. Never logged, never included in any to_dict() —
    # it exists only on this in-memory dataclass for the request's lifetime.
    api_key: str | None = None

    # The *deployment* name configured in the Azure OpenAI resource (not
    # the underlying base model name — Azure addresses deployments, not
    # models, in the Chat Completions call).
    deployment_name: str | None = None

    api_version: str = _DEFAULT_API_VERSION

    temperature: float = 0.2
    max_tokens: int = 700
    timeout_seconds: float = 30.0
    max_retries: int = 1

    # Payload-size guards: how much of each upstream report is embedded in
    # a prompt, so a run with thousands of groups still produces a
    # reasonably sized request.
    max_groups_in_prompt: int = 5
    max_important_features: int = 5
    max_rejected_models_in_prompt: int = 5

    # Whether enough credentials are present to attempt a call (all-or-nothing)
    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key and self.deployment_name)

    # Standard configuration: credentials from the environment
    @classmethod
    def default(cls) -> "LLMConfig":
        # A deployment with no Azure OpenAI access simply has these
        # environment variables unset; `is_configured` is then False and the
        # insight engine degrades gracefully rather than failing the run.
        return cls.from_env()

    # Build configuration from the standard Azure OpenAI environment variables
    @classmethod
    def from_env(cls) -> "LLMConfig":
        # Never hardcodes a credential — an unset variable simply leaves the
        # corresponding field `None`.
        return cls(
            endpoint=_env(AZURE_OPENAI_ENDPOINT_ENV_VAR),
            api_key=_env(AZURE_OPENAI_API_KEY_ENV_VAR),
            deployment_name=_env(AZURE_OPENAI_DEPLOYMENT_NAME_ENV_VAR),
            api_version=_env(AZURE_OPENAI_API_VERSION_ENV_VAR, _DEFAULT_API_VERSION),
        )

    # Build from a flat mapping, ignoring unknown keys
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LLMConfig":
        # Used for explicit overrides (tests, a caller that already resolved
        # its own settings) — default()/from_env() remain the normal path so
        # credentials are read from the environment, not passed as literals.
        if not payload:
            return cls()
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in allowed})


# Read an environment variable, treating blank as unset
def _env(name: str, default: str | None = None) -> str | None:
    # A .env file that lists optional Azure OpenAI settings as empty means
    # "not configured"; without this they would read as empty strings and
    # `is_configured` would still be False, but with a misleading value.
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default
