"""The engine reads its Azure OpenAI credentials from the Databricks scope.

The backend sends a scope NAME as a task parameter. No credential and no
{{secrets/...}} reference leaves it. The engine calls dbutils.secrets.get on
the cluster running the task, which is why both compute modes work the same:
a task parameter is identical for a job cluster and an existing one, while a
cluster environment variable exists only for a cluster this platform creates.

The template resolver is kept for the bundle job in databricks.yml, which
passes {{secrets/...}} on the --azure-openai-* flags.

Every test here uses a fake value; no real secret is ever read.
"""

from __future__ import annotations

import argparse
import sys
import types

import pytest

from forecast_engine.core import databricks_secrets
from forecast_engine.core.databricks_secrets import (
    SecretResolutionError,
    _resolve_databricks_secret_template,
    apply_azure_openai_cli_overrides,
    load_azure_openai_from_scope,
)

SECRET = "fake-secret-value-not-a-real-key"
REFERENCE = "{{secrets/forecastiq/azure-openai-api-key}}"


@pytest.fixture
def fake_dbutils(monkeypatch):
    """Stands in for the dbutils a Databricks cluster provides."""

    def install(getter):
        module = types.ModuleType("databricks.sdk.runtime")
        module.dbutils = types.SimpleNamespace(secrets=types.SimpleNamespace(get=getter))
        monkeypatch.setitem(sys.modules, "databricks.sdk.runtime", module)

    return install


def test_a_reference_resolves_to_the_secret_it_names(fake_dbutils):
    seen = {}

    def getter(scope, key):
        seen.update(scope=scope, key=key)
        return SECRET

    fake_dbutils(getter)

    assert _resolve_databricks_secret_template(REFERENCE) == SECRET
    assert seen == {"scope": "forecastiq", "key": "azure-openai-api-key"}


def test_a_plain_value_is_returned_untouched(fake_dbutils):
    """Local runs pass real values on the same flags; only the template shape
    is treated as a reference."""
    fake_dbutils(lambda scope, key: pytest.fail("must not consult secrets"))

    assert _resolve_databricks_secret_template("https://x.openai.azure.com/") == "https://x.openai.azure.com/"


# --- failing safely ----------------------------------------------------


def test_a_missing_secret_fails_with_a_message_naming_only_configuration(fake_dbutils):
    def getter(scope, key):
        raise RuntimeError("RESOURCE_DOES_NOT_EXIST")

    fake_dbutils(getter)

    with pytest.raises(SecretResolutionError) as raised:
        _resolve_databricks_secret_template(REFERENCE)

    message = str(raised.value)
    assert "azure-openai-api-key" in message and "forecastiq" in message
    assert SECRET not in message


def test_the_underlying_error_is_not_chained(fake_dbutils):
    """A client that renders its own call arguments must not reach a
    traceback — the cause is discarded, not chained."""

    def getter(scope, key):
        raise RuntimeError(f"failed calling get(scope={scope}, key={key}, value={SECRET})")

    fake_dbutils(getter)

    with pytest.raises(SecretResolutionError) as raised:
        _resolve_databricks_secret_template(REFERENCE)

    assert raised.value.__cause__ is None
    assert SECRET not in str(raised.value)


def test_an_empty_secret_is_an_error_not_a_blank_credential(fake_dbutils):
    fake_dbutils(lambda scope, key: "")

    with pytest.raises(SecretResolutionError, match="empty"):
        _resolve_databricks_secret_template(REFERENCE)


def test_off_databricks_says_so_rather_than_failing_obscurely(monkeypatch):
    def explode(name, *args, **kwargs):
        if name == "databricks.sdk.runtime":
            raise ImportError("no dbutils here")
        return original(name, *args, **kwargs)

    original = __import__
    monkeypatch.setattr("builtins.__import__", explode)

    with pytest.raises(SecretResolutionError, match="not.*running on Databricks"):
        _resolve_databricks_secret_template(REFERENCE)


# --- the flags the runner actually sends --------------------------------


def test_every_azure_flag_resolves_into_the_environment_the_engine_reads(fake_dbutils, monkeypatch):
    fake_dbutils(lambda scope, key: f"resolved::{key}")
    for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"):
        monkeypatch.delenv(var, raising=False)

    apply_azure_openai_cli_overrides(
        argparse.Namespace(
            azure_openai_endpoint="{{secrets/forecastiq/azure-openai-endpoint}}",
            azure_openai_api_key="{{secrets/forecastiq/azure-openai-api-key}}",
            azure_openai_deployment="{{secrets/forecastiq/azure-openai-deployment}}",
        )
    )

    import os

    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "resolved::azure-openai-endpoint"
    assert os.environ["AZURE_OPENAI_API_KEY"] == "resolved::azure-openai-api-key"
    assert os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] == "resolved::azure-openai-deployment"


def test_no_credential_is_written_to_output(monkeypatch):
    # The module logs one safe warning; it must never print, and the only
    # thing it logs is a SecretResolutionError, which carries no value.
    source = (databricks_secrets.__file__ or "").strip()
    assert source
    text = open(source).read()
    assert "print(" not in text
    assert 'logger.warning("LLM credentials unavailable: %s", exc)' in text


# --- reading straight from the scope ------------------------------------


def test_the_scope_name_alone_yields_every_credential(fake_dbutils, monkeypatch):
    # The whole simplification: the backend sends a scope name, nothing else.
    fake_dbutils(lambda scope, key: f"resolved::{scope}::{key}")
    for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"):
        monkeypatch.delenv(var, raising=False)

    load_azure_openai_from_scope("forecastiq")

    import os

    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "resolved::forecastiq::azure-openai-endpoint"
    assert os.environ["AZURE_OPENAI_API_KEY"] == "resolved::forecastiq::azure-openai-api-key"
    assert os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] == "resolved::forecastiq::azure-openai-deployment"


def test_no_scope_reads_nothing(fake_dbutils):
    fake_dbutils(lambda scope, key: pytest.fail("must not consult secrets"))

    load_azure_openai_from_scope("")
    load_azure_openai_from_scope(None)


def test_a_real_environment_variable_wins(fake_dbutils, monkeypatch):
    # Local runs set these directly and must not be overridden.
    fake_dbutils(lambda scope, key: "from-scope")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://local.openai.azure.com/")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    load_azure_openai_from_scope("forecastiq")

    import os

    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://local.openai.azure.com/"
    assert os.environ["AZURE_OPENAI_API_KEY"] == "from-scope"


def test_an_unreadable_scope_degrades_instead_of_failing_the_run(fake_dbutils, monkeypatch, caplog):
    # A missing insight must never cost a whole forecast.
    def getter(scope, key):
        raise RuntimeError(f"PERMISSION_DENIED value={SECRET}")

    fake_dbutils(getter)
    for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"):
        monkeypatch.delenv(var, raising=False)

    with caplog.at_level("WARNING"):
        load_azure_openai_from_scope("forecastiq")

    import os

    assert "AZURE_OPENAI_API_KEY" not in os.environ
    assert SECRET not in caplog.text
    assert "azure-openai" in caplog.text
