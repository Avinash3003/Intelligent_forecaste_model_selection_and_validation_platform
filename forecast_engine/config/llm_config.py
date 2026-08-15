"""Everything an Azure OpenAI call needs, read from the environment.

Azure OpenAI only — there is no provider abstraction. Prompt wording is a
separate concern and lives in s11_llm/prompts/.

Credentials are read once at construction and held in memory for the run —
never written to a log, a report, or the result.

Also carries the LLMOps knobs: token pricing, a per-run token budget, model
routing, and the fallback deployment. All default to unconfigured rather
than a guess, so a deployment degrades visibly instead of reporting a number
nobody set.
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

# Model routing (Section 13.2): a cheaper deployment for the common case —
# a single candidate that passed cleanly — and a stronger one reserved for
# the harder case, a decision with several rejected candidates to explain.
# Both are optional; an unset routing deployment simply means routing
# always uses the primary deployment above.
AZURE_OPENAI_DEPLOYMENT_SIMPLE_ENV_VAR = "AZURE_OPENAI_DEPLOYMENT_SIMPLE"
AZURE_OPENAI_DEPLOYMENT_COMPLEX_ENV_VAR = "AZURE_OPENAI_DEPLOYMENT_COMPLEX"

# Fallback provider (Section 11/13.2): a second Azure OpenAI deployment —
# a different resource, region, or SKU — tried only when the primary
# raises. Endpoint/key default to the primary's own when unset, so a
# deployment only needs to set a different deployment *name* to enable
# this on the same resource (e.g. a second, higher-quota deployment).
AZURE_OPENAI_FALLBACK_ENDPOINT_ENV_VAR = "AZURE_OPENAI_FALLBACK_ENDPOINT"
AZURE_OPENAI_FALLBACK_API_KEY_ENV_VAR = "AZURE_OPENAI_FALLBACK_API_KEY"
AZURE_OPENAI_FALLBACK_DEPLOYMENT_ENV_VAR = "AZURE_OPENAI_FALLBACK_DEPLOYMENT"

# Pricing (Section 8.5.2/13.2) — USD per 1,000 tokens. Never defaulted to a
# real-looking number: an unset price means cost is reported as
# unavailable, never fabricated. Set these from the Azure OpenAI resource's
# actual rate card for the deployed model/region.
AZURE_OPENAI_PRICE_INPUT_PER_1K_ENV_VAR = "AZURE_OPENAI_PRICE_INPUT_PER_1K"
AZURE_OPENAI_PRICE_OUTPUT_PER_1K_ENV_VAR = "AZURE_OPENAI_PRICE_OUTPUT_PER_1K"
AZURE_OPENAI_PRICE_INPUT_PER_1K_SIMPLE_ENV_VAR = "AZURE_OPENAI_PRICE_INPUT_PER_1K_SIMPLE"
AZURE_OPENAI_PRICE_OUTPUT_PER_1K_SIMPLE_ENV_VAR = "AZURE_OPENAI_PRICE_OUTPUT_PER_1K_SIMPLE"
AZURE_OPENAI_PRICE_INPUT_PER_1K_COMPLEX_ENV_VAR = "AZURE_OPENAI_PRICE_INPUT_PER_1K_COMPLEX"
AZURE_OPENAI_PRICE_OUTPUT_PER_1K_COMPLEX_ENV_VAR = "AZURE_OPENAI_PRICE_OUTPUT_PER_1K_COMPLEX"

# Section 13.2's per-run token ceiling. Generous enough that a normal-sized
# run never brushes it, low enough that a pathological run (thousands of
# keys, a retry storm) cannot run away unbounded.
LLM_MAX_TOKENS_PER_RUN_ENV_VAR = "LLM_MAX_TOKENS_PER_RUN"

# Azure OpenAI requires an explicit API version; this is the fallback used
# only when the environment variable above is not set.
_DEFAULT_API_VERSION = "2024-10-21"

# The active prompt version (Section 13.1 — "prompt versioning as a
# first-class artifact"). Bumping this is how a prompt-wording change is
# rolled forward, and pointing it back at an older value is how one is
# rolled back — both are a config change, not a code change. See
# `s11_llm/prompt_library.py` for how a version resolves to prompt files.
LLM_PROMPT_VERSION_ENV_VAR = "LLM_PROMPT_VERSION"
_DEFAULT_PROMPT_VERSION = "v2"


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
    max_tokens: int = 500
    timeout_seconds: float = 30.0
    max_retries: int = 1

    # Payload-size guards: how much of each upstream report is embedded in
    # a prompt, so a run with thousands of groups still produces a
    # reasonably sized request.
    max_groups_in_prompt: int = 5
    max_important_features: int = 5
    max_rejected_models_in_prompt: int = 5

    # --- LLMOps (Section 13) ------------------------------------------

    prompt_version: str = _DEFAULT_PROMPT_VERSION

    # Bounded retry count for a structured response that fails schema
    # validation (Section 13.1) — the failure is fed back into the retry
    # so the model sees exactly what was wrong, not just "try again".
    max_validation_retries: int = 2

    # Section 13.2's per-run ceiling. None (the default) means unbounded —
    # explicit opt-in, since an accidental low default would silently
    # truncate insights on a deployment that never asked for a budget.
    max_tokens_per_run: int | None = None

    # Model routing (Section 13.2). Both optional; a group's decision
    # routes to the simple deployment when it has zero rejected candidates
    # and to the complex one otherwise, falling back to `deployment_name`
    # for whichever tier has no deployment configured.
    deployment_name_simple: str | None = None
    deployment_name_complex: str | None = None

    # Fallback provider (Section 11/13.2). Endpoint/key fall back to the
    # primary's own when unset — only the deployment name is required to
    # enable this.
    fallback_endpoint: str | None = None
    fallback_api_key: str | None = None
    fallback_deployment_name: str | None = None

    # Pricing (Section 8.5.2/13.2), USD per 1,000 tokens. `None` means
    # "not configured" and must render as "cost unavailable", never $0 or a
    # guessed figure.
    price_input_per_1k: float | None = None
    price_output_per_1k: float | None = None
    price_input_per_1k_simple: float | None = None
    price_output_per_1k_simple: float | None = None
    price_input_per_1k_complex: float | None = None
    price_output_per_1k_complex: float | None = None

    # Whether enough credentials are present to attempt a call (all-or-nothing)
    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key and self.deployment_name)

    # Whether a distinct fallback deployment is usable
    @property
    def has_fallback(self) -> bool:
        return bool(self.fallback_deployment_name and (self.fallback_endpoint or self.endpoint) and (self.fallback_api_key or self.api_key))

    # Pricing for one routing tier, or the primary rate if the tier has none
    # configured. Returns (input_rate, output_rate); either element is None
    # when that side is not configured — never a fabricated number.
    def pricing_for(self, tier: str) -> tuple[float | None, float | None]:
        if tier == "simple" and self.price_input_per_1k_simple is not None:
            return self.price_input_per_1k_simple, self.price_output_per_1k_simple
        if tier == "complex" and self.price_input_per_1k_complex is not None:
            return self.price_input_per_1k_complex, self.price_output_per_1k_complex
        return self.price_input_per_1k, self.price_output_per_1k

    # Deployment name for one routing tier, falling back to the primary
    def deployment_for(self, tier: str) -> str | None:
        if tier == "simple" and self.deployment_name_simple:
            return self.deployment_name_simple
        if tier == "complex" and self.deployment_name_complex:
            return self.deployment_name_complex
        return self.deployment_name

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
            prompt_version=_env(LLM_PROMPT_VERSION_ENV_VAR, _DEFAULT_PROMPT_VERSION),
            max_tokens_per_run=_env_int(LLM_MAX_TOKENS_PER_RUN_ENV_VAR),
            deployment_name_simple=_env(AZURE_OPENAI_DEPLOYMENT_SIMPLE_ENV_VAR),
            deployment_name_complex=_env(AZURE_OPENAI_DEPLOYMENT_COMPLEX_ENV_VAR),
            fallback_endpoint=_env(AZURE_OPENAI_FALLBACK_ENDPOINT_ENV_VAR),
            fallback_api_key=_env(AZURE_OPENAI_FALLBACK_API_KEY_ENV_VAR),
            fallback_deployment_name=_env(AZURE_OPENAI_FALLBACK_DEPLOYMENT_ENV_VAR),
            price_input_per_1k=_env_float(AZURE_OPENAI_PRICE_INPUT_PER_1K_ENV_VAR),
            price_output_per_1k=_env_float(AZURE_OPENAI_PRICE_OUTPUT_PER_1K_ENV_VAR),
            price_input_per_1k_simple=_env_float(AZURE_OPENAI_PRICE_INPUT_PER_1K_SIMPLE_ENV_VAR),
            price_output_per_1k_simple=_env_float(AZURE_OPENAI_PRICE_OUTPUT_PER_1K_SIMPLE_ENV_VAR),
            price_input_per_1k_complex=_env_float(AZURE_OPENAI_PRICE_INPUT_PER_1K_COMPLEX_ENV_VAR),
            price_output_per_1k_complex=_env_float(AZURE_OPENAI_PRICE_OUTPUT_PER_1K_COMPLEX_ENV_VAR),
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


# Read an environment variable as a float, or None if unset/unparsable
def _env_float(name: str) -> float | None:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# Read an environment variable as an int, or None if unset/unparsable
def _env_int(name: str) -> int | None:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
