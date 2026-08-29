"""Why "I ran many pipelines but only see the latest one" happens, and the fix.

Root cause: `mlflow_tracking_uri` defaulted to a local sqlite file
regardless of execution mode. For `execution_mode=local` that is correct —
the engine subprocess runs on the same box and writes there too. For
`execution_mode=databricks` it is wrong: the engine runs as a
Databricks job and writes to the workspace's managed MLflow store, which
that sqlite file never touches. A backend that never separately set
MLFLOW_TRACKING_URI therefore read a store the engine never wrote to —
completed runs never appeared, so `list_runs()` fell back to whatever this
process still held in memory. Every restart (a redeploy, App Service
recycling) reset that to nothing, and the very last run submitted since
looked like "the only run".

A second, subtler half of the same bug: even once the tracking URI
resolves to "databricks", MLflow's own credential lookup reads
`os.environ` directly — never this app's Settings object — so a deployment
whose Databricks credentials arrived via a `.env` file (which
pydantic-settings loads into Settings without exporting to `os.environ`)
would still fail to authenticate, silently, with job submission
unaffected (it authenticates via an explicitly-built SDK Config instead).
"""

import os

import pytest

from app.config.settings import Settings
from app.orchestration.mlflow_history import MLflowHistoryStore


# ---------------------------------------------------------------------
# The tracking URI default is execution-mode aware
# ---------------------------------------------------------------------


def test_local_mode_still_defaults_to_the_shared_sqlite_file():
    """Unchanged: the engine subprocess and this backend share one box."""
    settings = Settings(execution_mode="local")
    assert settings.mlflow_tracking_uri_resolved.startswith("sqlite:///")


def test_databricks_mode_defaults_to_the_workspace_managed_store():
    settings = Settings(execution_mode="databricks")
    assert settings.mlflow_tracking_uri_resolved == "databricks"



def test_an_explicitly_configured_uri_is_never_overridden():
    """Configured values are used as-is regardless of execution mode —
    an operator who deliberately pointed at something else is respected."""
    settings = Settings(execution_mode="databricks", mlflow_tracking_uri="sqlite:////tmp/custom.db")
    assert settings.mlflow_tracking_uri_resolved == "sqlite:////tmp/custom.db"


def test_mode_matching_is_case_and_whitespace_insensitive():
    settings = Settings(execution_mode="  DataBricks ")
    assert settings.mlflow_tracking_uri_resolved == "databricks"


# ---------------------------------------------------------------------
# Databricks credentials reach MLflow's own (separate) auth lookup
# ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_databricks_env(monkeypatch):
    """Every test starts with none of these set, and cleans up after —
    they are real process environment variables the fix writes to, and
    must never leak between tests or pollute the real dev environment."""
    for name in ("DATABRICKS_HOST", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET", "DATABRICKS_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_databricks_credentials_are_exported_for_mlflows_own_lookup():
    """The scenario this exists for: credentials reached Settings (e.g.
    via a .env file) but were never real environment variables — MLflow's
    databricks auth reads os.environ directly and would otherwise find
    nothing, regardless of what job submission's own SDK client used."""
    settings = Settings(
        execution_mode="databricks",
        databricks_host="https://example.azuredatabricks.net",
        databricks_client_id="client-123",
        databricks_client_secret="secret-456",
    )
    store = MLflowHistoryStore(settings)
    store._ensure_databricks_credentials_in_environment()

    assert os.environ["DATABRICKS_HOST"] == "https://example.azuredatabricks.net"
    assert os.environ["DATABRICKS_CLIENT_ID"] == "client-123"
    assert os.environ["DATABRICKS_CLIENT_SECRET"] == "secret-456"


def test_the_legacy_token_credential_is_also_exported():
    settings = Settings(
        execution_mode="databricks",
        databricks_host="https://example.azuredatabricks.net",
        databricks_token="dapi-abc",
    )
    store = MLflowHistoryStore(settings)
    store._ensure_databricks_credentials_in_environment()

    assert os.environ["DATABRICKS_TOKEN"] == "dapi-abc"


def test_an_operators_own_real_environment_variable_is_never_overridden(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://the-real-workspace.azuredatabricks.net")
    settings = Settings(execution_mode="databricks", databricks_host="https://settings-value.azuredatabricks.net")

    store = MLflowHistoryStore(settings)
    store._ensure_databricks_credentials_in_environment()

    assert os.environ["DATABRICKS_HOST"] == "https://the-real-workspace.azuredatabricks.net"


def test_nothing_is_exported_for_a_non_databricks_tracking_uri():
    settings = Settings(execution_mode="local", databricks_host="https://example.azuredatabricks.net")
    store = MLflowHistoryStore(settings)
    store._ensure_databricks_credentials_in_environment()

    assert "DATABRICKS_HOST" not in os.environ


def test_no_credentials_configured_exports_nothing_and_does_not_raise():
    settings = Settings(execution_mode="databricks")
    store = MLflowHistoryStore(settings)
    store._ensure_databricks_credentials_in_environment()  # must not raise

    assert "DATABRICKS_CLIENT_ID" not in os.environ


# ---------------------------------------------------------------------
# /health surfaces the reachability this bug hid
# ---------------------------------------------------------------------


def test_health_reports_which_run_history_backend_is_configured(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert "run_history_available" in body
        assert "run_history_backend" in body
        assert "execution_backend" in body


def test_health_never_leaks_the_full_tracking_uri():
    """The URI can carry embedded credentials on some deployments; /health
    is unauthenticated, so only the scheme is ever reported."""
    from fastapi.testclient import TestClient

    from app.config.settings import get_settings
    from app.main import app

    app.dependency_overrides[get_settings] = lambda: Settings(
        mlflow_tracking_uri="postgresql://user:supersecret@db-host/mlflow"
    )
    try:
        with TestClient(app) as client:
            body = client.get("/health").json()
            assert "supersecret" not in str(body)
            assert body["run_history_backend"] == "postgresql"
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------
# OAuth needs one more environment variable than the credentials
# ---------------------------------------------------------------------


def test_oauth_credentials_also_enable_mlflows_databricks_sdk_path(monkeypatch):
    """mlflow-skinny refuses OAuth outright without this.

    Handed a client id and secret and no token, `http_request` raises
    "To use OAuth authentication, set environmental variable
    'MLFLOW_ENABLE_DB_SDK' to true" on every call — so run history reads
    as empty rather than as broken, since building the client is not what
    fails. This deployment has no personal access token by design.
    """
    monkeypatch.delenv("MLFLOW_ENABLE_DB_SDK", raising=False)
    settings = Settings(
        execution_mode="databricks",
        databricks_host="https://example.azuredatabricks.net",
        databricks_client_id="client-123",
        databricks_client_secret="secret-456",
    )

    MLflowHistoryStore(settings)._ensure_databricks_credentials_in_environment()

    assert os.environ["MLFLOW_ENABLE_DB_SDK"] == "true"


def test_an_operator_who_disabled_the_sdk_path_is_not_overridden(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_DB_SDK", "false")
    settings = Settings(
        execution_mode="databricks",
        databricks_host="https://example.azuredatabricks.net",
        databricks_client_id="client-123",
        databricks_client_secret="secret-456",
    )

    MLflowHistoryStore(settings)._ensure_databricks_credentials_in_environment()

    assert os.environ["MLFLOW_ENABLE_DB_SDK"] == "false"


def test_token_only_credentials_do_not_set_the_oauth_flag(monkeypatch):
    """A token authenticates on its own; the flag would be noise."""
    monkeypatch.delenv("MLFLOW_ENABLE_DB_SDK", raising=False)
    settings = Settings(
        execution_mode="databricks",
        databricks_host="https://example.azuredatabricks.net",
        databricks_token="dapi-abc",
    )

    MLflowHistoryStore(settings)._ensure_databricks_credentials_in_environment()

    assert "MLFLOW_ENABLE_DB_SDK" not in os.environ
