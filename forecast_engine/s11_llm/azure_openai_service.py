"""The platform's only LLM implementation.

No provider interface or registry: with a single implementation an
abstraction would add indirection without buying flexibility.

The SDK is imported lazily inside complete(), like every other optional
dependency here, so a deployment that never enables insights need not
install it and a missing package degrades to "unavailable".

complete() returns a result object rather than a bare string because the
caller needs to know what one call cost — the SDK response carries usage,
and nothing upstream could reconstruct it afterwards.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from forecast_engine.config.llm_config import LLMConfig
from forecast_engine.utils.exceptions import LLMProviderError


@dataclass
class LLMCompletionResult:
    """One call's text plus everything Section 13's telemetry needs."""

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float


class AzureOpenAIService:
    """Calls an Azure OpenAI Chat Completions deployment.

    Holds up to two client configurations — primary and fallback (Section
    11/13.2) — each lazily constructed and cached on first use, so a
    deployment that never needs the fallback pays nothing for it.

    One instance is shared by every insight worker, so construction is
    guarded; the SDK client itself is safe to call concurrently.
    """

    # Store config; clients are constructed lazily
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = None
        self._fallback_client = None
        self._client_lock = threading.Lock()

    # Whether a call can be attempted: credentials present and openai importable
    def is_available(self, *, use_fallback: bool = False) -> bool:
        if use_fallback:
            if not self._config.has_fallback:
                return False
        elif not self._config.is_configured:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    # Human-readable reason is_available() is False
    def unavailable_reason(self, *, use_fallback: bool = False) -> str:
        # Checked in the same order is_available() checks, so the first
        # reported reason is always the first real blocker.
        if use_fallback:
            if not self._config.fallback_deployment_name:
                return "No fallback deployment is configured."
        else:
            if not self._config.endpoint:
                return "AZURE_OPENAI_ENDPOINT is not set."
            if not self._config.api_key:
                return "AZURE_OPENAI_API_KEY is not set."
            if not self._config.deployment_name:
                return "AZURE_OPENAI_DEPLOYMENT_NAME is not set."
        try:
            import openai  # noqa: F401
        except ImportError:
            return "The 'openai' package is not installed."
        return ""

    # Return one deployment's response for one prompt; never raises a raw SDK exception
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        deployment: str | None = None,
        use_fallback: bool = False,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> LLMCompletionResult:
        """Call `deployment` (or the routing/primary default) and return
        its text plus token/latency telemetry.

        Args:
            deployment: Overrides which Azure deployment name is called —
                how model routing (Section 13.2) reaches a specific tier
                without this class knowing routing exists.
            use_fallback: Use the fallback endpoint/key (Section 11) rather
                than the primary. `deployment` still overrides the
                deployment name within whichever client is selected.
            json_mode: Request the SDK's structured-output mode when the
                deployment supports it — a defense-in-depth complement to
                the prompt's own instruction to return bare JSON, not a
                replacement for `schema.parse_and_validate`.
            max_tokens: Overrides `config.max_tokens` for this call.
        """
        if not self.is_available(use_fallback=use_fallback):
            raise LLMProviderError(self.unavailable_reason(use_fallback=use_fallback))

        try:
            from openai import (
                APIConnectionError,
                APIError,
                APITimeoutError,
                AuthenticationError,
                AzureOpenAI,
                BadRequestError,
                RateLimitError,
            )
        except ImportError as exc:
            raise LLMProviderError(
                "The 'openai' package is not installed; run `pip install openai` to enable "
                "Azure OpenAI business insights."
            ) from exc

        resolved_deployment = deployment or (
            self._config.fallback_deployment_name if use_fallback else self._config.deployment_name
        )
        started = time.perf_counter()

        try:
            client = self._get_client(AzureOpenAI, use_fallback=use_fallback)
            kwargs: dict = {
                "model": resolved_deployment,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self._config.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self._config.max_tokens,
            }
            if json_mode:
                # Best-effort: older deployments/models reject this
                # parameter, in which case the caller's own prompt
                # instruction (and schema.parse_and_validate afterward) is
                # what actually enforces the contract.
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
        except AuthenticationError as exc:
            raise LLMProviderError(f"Azure OpenAI authentication failed: {exc}") from exc
        except RateLimitError as exc:
            raise LLMProviderError(f"Azure OpenAI rate limit exceeded: {exc}") from exc
        except APITimeoutError as exc:
            raise LLMProviderError(f"Azure OpenAI request timed out: {exc}") from exc
        except APIConnectionError as exc:
            raise LLMProviderError(f"Could not reach Azure OpenAI: {exc}") from exc
        except BadRequestError as exc:
            if json_mode:
                # Retry once without response_format — some deployments
                # (older API versions, some model families) reject the
                # parameter outright rather than ignoring it.
                return self.complete(
                    system_prompt, user_prompt, deployment=deployment, use_fallback=use_fallback,
                    json_mode=False, max_tokens=max_tokens,
                )
            raise LLMProviderError(f"Azure OpenAI rejected the request: {exc}") from exc
        except APIError as exc:
            # Base class for every other Azure/OpenAI API error — caught
            # last so the more specific handlers above take precedence.
            raise LLMProviderError(f"Azure OpenAI API error: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            text = str(response.choices[0].message.content).strip()
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"Azure OpenAI response was not in the expected shape: {exc}") from exc

        usage = getattr(response, "usage", None)
        return LLMCompletionResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            total_tokens=getattr(usage, "total_tokens", None) if usage else None,
            latency_ms=latency_ms,
        )

    # Construct (once) and cache the Azure OpenAI client for the requested side
    def _get_client(self, azure_openai_cls: type, *, use_fallback: bool) -> object:
        # Building the client only validates arguments locally (no network call),
        # so constructing it lazily here keeps import/construction free of I/O.
        with self._client_lock:
            if use_fallback:
                if self._fallback_client is None:
                    self._fallback_client = azure_openai_cls(
                        azure_endpoint=self._config.fallback_endpoint or self._config.endpoint,
                        api_key=self._config.fallback_api_key or self._config.api_key,
                        api_version=self._config.api_version,
                        timeout=self._config.timeout_seconds,
                        max_retries=self._config.max_retries,
                    )
                return self._fallback_client

            if self._client is None:
                self._client = azure_openai_cls(
                    azure_endpoint=self._config.endpoint,
                    api_key=self._config.api_key,
                    api_version=self._config.api_version,
                    timeout=self._config.timeout_seconds,
                    max_retries=self._config.max_retries,
                )
            return self._client
