"""Checks whether the LLM is configured and reachable.

Two deliberately separate checks:
  - check_configuration: static, no network call. Reports whether each
    setting is present, never its value, so it is safe to log or return.
  - run_connectivity_test: exactly one request, the smallest that proves
    the whole path works (auth, endpoint, deployment, telemetry) without the
    cost of a real insight prompt.

Neither belongs in the insight engine, whose job is turning a run into
insights — not answering "is this deployment reachable".
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from forecast_engine.config.llm_config import LLMConfig
from forecast_engine.s11_llm.azure_openai_service import AzureOpenAIService


@dataclass
class LLMConfigurationCheck:
    """Presence, not value, of every setting an Azure OpenAI call needs."""

    enabled: bool
    endpoint_configured: bool
    deployment_configured: bool
    authentication_method: str  # "api_key" | "none"
    credentials_available: bool
    api_version: str
    prompt_version: str
    fallback_configured: bool
    routing_configured: bool
    pricing_configured: bool
    ready: bool  # everything a call needs is present

    def to_dict(self) -> dict:
        return asdict(self)


def check_configuration(config: LLMConfig | None = None) -> LLMConfigurationCheck:
    """Inspect an `LLMConfig` and report what is set, without exposing it.

    `authentication_method` is always "api_key" or "none": the SDK client
    this platform builds (`AzureOpenAIService._get_client`) only ever
    constructs an `AzureOpenAI(api_key=...)` client — there is no
    Entra/AAD-token code path today, so reporting anything else here would
    describe a mechanism that does not exist.
    """
    config = config or LLMConfig.default()

    return LLMConfigurationCheck(
        enabled=config.enabled,
        endpoint_configured=bool(config.endpoint),
        deployment_configured=bool(config.deployment_name),
        authentication_method="api_key" if config.api_key else "none",
        credentials_available=bool(config.api_key),
        api_version=config.api_version,
        prompt_version=config.prompt_version,
        fallback_configured=config.has_fallback,
        routing_configured=bool(config.deployment_name_simple or config.deployment_name_complex),
        pricing_configured=config.price_input_per_1k is not None and config.price_output_per_1k is not None,
        ready=config.enabled and config.is_configured,
    )


@dataclass
class LLMConnectivityResult:
    """Outcome of one minimal, real Azure OpenAI request."""

    success: bool
    deployment: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float | None
    used_fallback: bool
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def run_connectivity_test(
    config: LLMConfig | None = None, service: AzureOpenAIService | None = None
) -> LLMConnectivityResult:
    """Make exactly one real Azure OpenAI request and report the outcome.

    Deliberately not a business-insight prompt — a fixed, tiny system/user
    pair that costs a handful of tokens and proves the same path a real
    call uses (client construction, deployment resolution, the SDK
    request, token/latency capture) without generating anything the
    platform would treat as a real insight. Tries the fallback only if the
    primary is unavailable, mirroring `LLMInsightEngine`'s own order.

    Never raises: a connectivity problem is exactly what this reports,
    not what it propagates.
    """
    config = config or LLMConfig.default()
    service = service or AzureOpenAIService(config)

    use_fallback = not service.is_available() and service.is_available(use_fallback=True)
    if not service.is_available(use_fallback=use_fallback):
        return LLMConnectivityResult(
            success=False,
            deployment=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_ms=None,
            used_fallback=use_fallback,
            error=service.unavailable_reason(use_fallback=use_fallback),
        )

    started = time.perf_counter()
    try:
        result = service.complete(
            system_prompt="You are a connectivity check. Reply with exactly one word.",
            user_prompt="Reply with the single word: ok",
            use_fallback=use_fallback,
            max_tokens=5,
        )
    except Exception as exc:  # noqa: BLE001 - every provider error becomes one reported outcome
        return LLMConnectivityResult(
            success=False,
            deployment=config.fallback_deployment_name if use_fallback else config.deployment_name,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            used_fallback=use_fallback,
            error=str(exc),
        )

    return LLMConnectivityResult(
        success=True,
        deployment=config.fallback_deployment_name if use_fallback else config.deployment_name,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        latency_ms=result.latency_ms,
        used_fallback=use_fallback,
        error=None,
    )
