"""Azure OpenAI service — the platform's sole LLM implementation
(Section 6.12, "Model Provider").

There is deliberately no provider interface or registry here: the platform
uses Azure OpenAI only, so an abstraction with a single implementation
would add indirection without buying flexibility. `LLMInsightEngine`
depends on this concrete class directly.

The Azure OpenAI SDK is imported lazily, inside `complete()`, mirroring the
lazy-optional-dependency pattern used everywhere else in the engine
(SHAP, Prophet, TFT): a deployment that never enables LLM insights is not
forced to install it, and a missing package degrades to an "unavailable"
report rather than an import-time crash.
"""

from __future__ import annotations

from typing import Any

from forecast_engine.config.llm_config import LLMConfig
from forecast_engine.utils.exceptions import LLMProviderError


class AzureOpenAIService:
    """Calls one Azure OpenAI Chat Completions deployment."""

    # Store config; client is constructed lazily
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = None  # constructed lazily on first use

    # Whether a call can be attempted: credentials present and openai importable
    def is_available(self) -> bool:
        if not self._config.is_configured:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    # Human-readable reason is_available() is False
    def unavailable_reason(self) -> str:
        # Checked in the same order is_available() checks, so the first
        # reported reason is always the first real blocker.
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

    # Return the deployment's response text for one prompt; never raises a raw SDK exception
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self._config.is_configured:
            raise LLMProviderError(self.unavailable_reason())

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

        try:
            client = self._get_client(AzureOpenAI)
            response = client.chat.completions.create(
                model=self._config.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
        except AuthenticationError as exc:
            raise LLMProviderError(f"Azure OpenAI authentication failed: {exc}") from exc
        except RateLimitError as exc:
            raise LLMProviderError(f"Azure OpenAI rate limit exceeded: {exc}") from exc
        except APITimeoutError as exc:
            raise LLMProviderError(f"Azure OpenAI request timed out: {exc}") from exc
        except APIConnectionError as exc:
            raise LLMProviderError(f"Could not reach Azure OpenAI: {exc}") from exc
        except BadRequestError as exc:
            raise LLMProviderError(f"Azure OpenAI rejected the request: {exc}") from exc
        except APIError as exc:
            # Base class for every other Azure/OpenAI API error — caught
            # last so the more specific handlers above take precedence.
            raise LLMProviderError(f"Azure OpenAI API error: {exc}") from exc

        try:
            return str(response.choices[0].message.content).strip()
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"Azure OpenAI response was not in the expected shape: {exc}") from exc

    # Construct (once) and cache the Azure OpenAI client
    def _get_client(self, azure_openai_cls: type) -> Any:
        # Building the client only validates arguments locally (no network call),
        # so constructing it lazily here keeps import/construction free of I/O.
        if self._client is None:
            self._client = azure_openai_cls(
                azure_endpoint=self._config.endpoint,
                api_key=self._config.api_key,
                api_version=self._config.api_version,
                timeout=self._config.timeout_seconds,
                max_retries=self._config.max_retries,
            )
        return self._client
