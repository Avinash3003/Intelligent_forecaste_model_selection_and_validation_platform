"""LLM configuration and connectivity diagnostics.

Production cloud runs currently show provider=template, call_count=0 —
the jobs.submit path never passes Azure OpenAI credentials to the wheel
task. These diagnostics exist so that gap is visible on its own terms
(a config check that never touches the network) and so a single real
request can prove the deployment credentials actually work, independent
of which compute backend eventually calls them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


from forecast_engine.config.llm_config import LLMConfig
from forecast_engine.s11_llm.azure_openai_service import LLMCompletionResult
from forecast_engine.s11_llm.diagnostics import check_configuration, run_connectivity_test


# ---------------------------------------------------------------------
# check_configuration — static, no network, no secret values
# ---------------------------------------------------------------------


def test_a_fully_configured_deployment_is_reported_ready():
    config = LLMConfig(endpoint="https://x.openai.azure.com/", api_key="secret", deployment_name="gpt-4")

    check = check_configuration(config)

    assert check.ready is True
    assert check.endpoint_configured is True
    assert check.deployment_configured is True
    assert check.credentials_available is True
    assert check.authentication_method == "api_key"


def test_an_unconfigured_deployment_is_reported_not_ready():
    check = check_configuration(LLMConfig(endpoint=None, api_key=None, deployment_name=None))

    assert check.ready is False
    assert check.endpoint_configured is False
    assert check.credentials_available is False
    assert check.authentication_method == "none"


def test_partial_configuration_is_reported_precisely():
    """Exactly what production shows: nothing reaches the wheel task, so
    every field must read as absent, not as a generic 'not ready'."""
    check = check_configuration(LLMConfig(endpoint=None, api_key=None, deployment_name=None))

    assert check.endpoint_configured is False
    assert check.deployment_configured is False
    assert check.credentials_available is False


def test_the_check_never_includes_the_actual_credential_value():
    config = LLMConfig(
        endpoint="https://x.openai.azure.com/", api_key="super-secret-value", deployment_name="gpt-4"
    )

    check = check_configuration(config)
    serialized = str(check.to_dict())

    assert "super-secret-value" not in serialized


def test_disabled_is_reported_as_not_ready_even_if_otherwise_complete():
    config = LLMConfig(
        enabled=False, endpoint="https://x.openai.azure.com/", api_key="k", deployment_name="gpt-4"
    )

    assert check_configuration(config).ready is False


def test_fallback_and_routing_and_pricing_are_reported_independently():
    config = LLMConfig(
        endpoint="https://x.openai.azure.com/",
        api_key="k",
        deployment_name="gpt-4",
        fallback_deployment_name="gpt-4-fallback",
        deployment_name_simple="gpt-4-mini",
        price_input_per_1k=0.15,
        price_output_per_1k=0.6,
    )

    check = check_configuration(config)

    assert check.fallback_configured is True
    assert check.routing_configured is True
    assert check.pricing_configured is True


def test_pricing_requires_both_input_and_output_rate():
    config = LLMConfig(
        endpoint="https://x.openai.azure.com/", api_key="k", deployment_name="gpt-4",
        price_input_per_1k=0.15, price_output_per_1k=None,
    )

    assert check_configuration(config).pricing_configured is False


# ---------------------------------------------------------------------
# run_connectivity_test — exactly one call, real service interface
# ---------------------------------------------------------------------


@dataclass
class _FakeService:
    """Duck-typed `AzureOpenAIService`, scripted for one call."""

    available: bool = True
    fallback_available: bool = False
    response: LLMCompletionResult | Exception | None = None
    calls: list = field(default_factory=list)

    def is_available(self, *, use_fallback: bool = False) -> bool:
        return self.fallback_available if use_fallback else self.available

    def unavailable_reason(self, *, use_fallback: bool = False) -> str:
        return "AZURE_OPENAI_ENDPOINT is not set."

    def complete(self, system_prompt, user_prompt, *, use_fallback=False, max_tokens=None, **_):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, "use_fallback": use_fallback})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_a_successful_connectivity_test_reports_tokens_and_latency():
    service = _FakeService(response=LLMCompletionResult(text="ok", prompt_tokens=12, completion_tokens=1, total_tokens=13, latency_ms=250.0))
    config = LLMConfig(endpoint="https://x/", api_key="k", deployment_name="gpt-4")

    result = run_connectivity_test(config, service)

    assert result.success is True
    assert result.deployment == "gpt-4"
    assert result.total_tokens == 13
    assert result.latency_ms == 250.0
    assert result.error is None
    assert result.used_fallback is False


def test_exactly_one_call_is_made():
    service = _FakeService(response=LLMCompletionResult(text="ok", prompt_tokens=1, completion_tokens=1, total_tokens=2, latency_ms=1.0))

    run_connectivity_test(LLMConfig(endpoint="https://x/", api_key="k", deployment_name="gpt-4"), service)

    assert len(service.calls) == 1


def test_an_unconfigured_deployment_is_reported_without_attempting_a_call():
    service = _FakeService(available=False)

    result = run_connectivity_test(LLMConfig(endpoint=None, api_key=None, deployment_name=None), service)

    assert result.success is False
    assert result.error
    assert service.calls == []


def test_a_provider_error_is_reported_not_raised():
    service = _FakeService(response=RuntimeError("Azure OpenAI authentication failed"))

    result = run_connectivity_test(LLMConfig(endpoint="https://x/", api_key="bad", deployment_name="gpt-4"), service)

    assert result.success is False
    assert "authentication failed" in result.error
    assert result.latency_ms is not None


def test_falls_back_when_the_primary_is_unavailable_but_the_fallback_is_configured():
    service = _FakeService(
        available=False,
        fallback_available=True,
        response=LLMCompletionResult(text="ok", prompt_tokens=1, completion_tokens=1, total_tokens=2, latency_ms=1.0),
    )
    config = LLMConfig(
        endpoint=None, api_key=None, deployment_name=None,
        fallback_endpoint="https://y/", fallback_api_key="k2", fallback_deployment_name="gpt-4-fb",
    )

    result = run_connectivity_test(config, service)

    assert result.success is True
    assert result.used_fallback is True
    assert result.deployment == "gpt-4-fb"


def test_neither_primary_nor_fallback_available_reports_failure_without_a_call():
    service = _FakeService(available=False, fallback_available=False)

    result = run_connectivity_test(LLMConfig(endpoint=None, api_key=None, deployment_name=None), service)

    assert result.success is False
    assert service.calls == []
