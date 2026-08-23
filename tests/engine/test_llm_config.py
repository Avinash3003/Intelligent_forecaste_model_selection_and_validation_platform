"""LLMConfig's LLMOps knobs: pricing, budget, routing, fallback."""

from forecast_engine.config.llm_config import LLMConfig


def test_unconfigured_pricing_is_none_never_fabricated():
    config = LLMConfig()
    assert config.pricing_for("simple") == (None, None)


def test_configured_primary_pricing_applies_to_both_tiers_by_default():
    config = LLMConfig(price_input_per_1k=0.15, price_output_per_1k=0.6)
    assert config.pricing_for("simple") == (0.15, 0.6)
    assert config.pricing_for("complex") == (0.15, 0.6)


def test_tier_specific_pricing_overrides_the_primary_rate():
    config = LLMConfig(
        price_input_per_1k=0.15, price_output_per_1k=0.6,
        price_input_per_1k_complex=0.5, price_output_per_1k_complex=1.5,
    )
    assert config.pricing_for("simple") == (0.15, 0.6)
    assert config.pricing_for("complex") == (0.5, 1.5)


def test_deployment_for_tier_falls_back_to_primary_when_unset():
    config = LLMConfig(deployment_name="gpt-4o-mini")
    assert config.deployment_for("simple") == "gpt-4o-mini"
    assert config.deployment_for("complex") == "gpt-4o-mini"


def test_deployment_for_tier_uses_configured_routing_deployment():
    config = LLMConfig(
        deployment_name="gpt-4o-mini", deployment_name_simple="gpt-35-turbo", deployment_name_complex="gpt-4o",
    )
    assert config.deployment_for("simple") == "gpt-35-turbo"
    assert config.deployment_for("complex") == "gpt-4o"


def test_has_fallback_requires_a_fallback_deployment():
    assert not LLMConfig(endpoint="https://x", api_key="k").has_fallback
    assert LLMConfig(
        endpoint="https://x", api_key="k", fallback_deployment_name="backup"
    ).has_fallback


def test_fallback_reuses_primary_endpoint_and_key_when_unset():
    config = LLMConfig(endpoint="https://x", api_key="primary-key", fallback_deployment_name="backup")
    assert config.has_fallback


def test_max_tokens_per_run_defaults_to_unbounded():
    assert LLMConfig().max_tokens_per_run is None


def test_from_env_reads_llmops_settings(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    monkeypatch.setenv("LLM_PROMPT_VERSION", "v2")
    monkeypatch.setenv("LLM_MAX_TOKENS_PER_RUN", "50000")
    monkeypatch.setenv("AZURE_OPENAI_PRICE_INPUT_PER_1K", "0.15")
    monkeypatch.setenv("AZURE_OPENAI_PRICE_OUTPUT_PER_1K", "0.6")

    config = LLMConfig.from_env()
    assert config.is_configured
    assert config.prompt_version == "v2"
    assert config.max_tokens_per_run == 50000
    assert config.price_input_per_1k == 0.15
    assert config.price_output_per_1k == 0.6


def test_from_env_treats_blank_string_as_unset(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "   ")
    config = LLMConfig.from_env()
    assert config.endpoint is None


def test_malformed_numeric_env_vars_do_not_crash(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS_PER_RUN", "not-a-number")
    monkeypatch.setenv("AZURE_OPENAI_PRICE_INPUT_PER_1K", "also-not-a-number")
    config = LLMConfig.from_env()
    assert config.max_tokens_per_run is None
    assert config.price_input_per_1k is None
