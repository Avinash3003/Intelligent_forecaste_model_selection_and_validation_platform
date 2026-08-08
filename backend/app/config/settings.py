import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration. No secrets or hosts are hardcoded —
    values default to local/dummy-friendly settings and are overridden via
    a .env file or real environment variables in each deployment target."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Forecast IQ API"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    debug: bool = True

    cors_origins: list[str] = ["http://localhost:5173"]

    upload_dir: str = "uploads"
    max_upload_size_mb: int = 200

    # Pipeline Orchestration & Execution Layer (Section 6.14). "local" runs
    # the forecast_engine pipeline as a local subprocess; "databricks" would
    # run it as a Databricks Job (architecture only in this phase — see
    # app/orchestration/databricks_runner.py). Retargeting environments is
    # this one value changing, never a code change.
    execution_mode: str = "local"

    # Where the standalone forecast_engine package (and its own venv) live,
    # relative to the backend process's working directory. Never a
    # hardcoded absolute path, so the same code runs from any checkout.
    forecast_engine_root: str = "../forecast_engine"

    # Override for the forecast_engine interpreter; when unset, it is
    # derived from `forecast_engine_root` (see `forecast_engine_python_path`).
    forecast_engine_python: str | None = None

    job_poll_interval_seconds: float = 2.0
    job_timeout_seconds: float = 3600.0

    # forecast_engine's LLM Insight Engine (Azure OpenAI, Section 6.12) and
    # MLflow tracking layer (Section 6.13) read these exact variable names
    # directly from *their own* process environment — forecast_engine runs
    # as a subprocess LocalRunner launches (see app/orchestration/
    # local_runner.py), not as code imported into this process. Declaring
    # them here makes this backend's `.env` the one place a developer edits
    # them; `subprocess_env()` below is what actually forwards them into
    # the subprocess so that single `.env` file is genuinely sufficient —
    # pydantic-settings loading a `.env` file does NOT, by itself, export
    # those values into `os.environ` for a child process to inherit.
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None

    mlflow_tracking_uri: str | None = None
    mlflow_registry_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_artifact_location: str | None = None

    # Reserved for the Azure deployment phase — unused until Databricks/
    # Azure Storage integration is actually implemented (see
    # app/orchestration/databricks_runner.py). Left unset by default; no
    # dummy values are treated as real, and nothing reads these yet.
    databricks_host: str | None = None
    databricks_token: str | None = None
    databricks_workspace_id: str | None = None

    azure_storage_connection_string: str | None = None
    # Dataset preview reads the uploaded file back from ADLS. A container-
    # scoped read-only SAS is preferred over the connection string: it cannot
    # write and it expires, so the backend never holds account-wide rights.
    azure_storage_account: str | None = None
    azure_storage_sas_token: str | None = None
    azure_uploads_container: str = "uploads"
    azure_tenant_id: str | None = None
    azure_client_id: str | None = None
    azure_client_secret: str | None = None

    @property
    def upload_path(self) -> Path:
        # Resolved to an absolute path: the staged dataset path is handed to
        # the forecast_engine subprocess, which runs with a different working
        # directory (the project root), so a relative path would not resolve
        # there.
        path = Path(self.upload_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def engine_working_dir(self) -> Path:
        """Working directory the forecast_engine subprocess runs in.

        `-m forecast_engine.run_pipeline` resolves the top-level package
        from the current directory, so the subprocess runs from
        forecast_engine's *parent*. Every relative path the engine writes —
        most importantly its default local MLflow store — lands here, which
        is why this backend must resolve those same paths against this
        directory rather than its own.
        """
        return Path(self.forecast_engine_root).resolve().parent

    @property
    def mlflow_tracking_uri_resolved(self) -> str:
        """The tracking URI this backend reads run history from.

        Must name the same store the engine writes to. When configured
        (a remote server, or Databricks) that value is used verbatim and
        both sides agree trivially. When unset, the engine falls back to
        its local default of `sqlite:///mlflow.db` — a *relative* URI,
        which would otherwise resolve to this process's own directory
        instead of the subprocess's. Absolutizing it against
        `engine_working_dir` is what keeps the two pointing at one file.
        """
        configured = (self.mlflow_tracking_uri or "").strip()
        if configured:
            return configured
        return f"sqlite:///{self.engine_working_dir / 'mlflow.db'}"

    @property
    def mlflow_experiment_name_resolved(self) -> str:
        # Mirrors forecast_engine/config/mlflow_config.py's own default, so
        # an unconfigured deployment still reads the experiment the engine
        # wrote to.
        return (self.mlflow_experiment_name or "").strip() or "/forecast-engine"

    @property
    def forecast_engine_python_path(self) -> Path:
        """Interpreter LocalRunner invokes forecast_engine with.

        Defaults to the standalone package's own venv — forecast_engine
        carries heavy, potentially conflicting dependencies (xgboost,
        statsmodels, mlflow, ...), so it is run as a separate process with
        its own interpreter rather than imported into this process.

        Always absolute: LocalRunner launches the subprocess with its `cwd`
        set to the forecast_engine directory, and a relative path here
        would then be resolved against *that* directory a second time
        instead of the backend's own working directory.

        Absolutized with `os.path.abspath`, not `Path.resolve()` —
        `.venv/bin/python` is itself a symlink to the base interpreter, and
        `resolve()` follows it all the way through to e.g. `/usr/bin/
        python3.12`. Executing that resolved target directly loses venv
        activation (Python locates a venv's `pyvenv.cfg` from the path it
        was *invoked* with, not the symlink's ultimate target), which
        silently drops forecast_engine's installed dependencies.
        """
        if self.forecast_engine_python:
            candidate = Path(self.forecast_engine_python)
        else:
            candidate = Path(self.forecast_engine_root) / ".venv" / "bin" / "python"
        return candidate if candidate.is_absolute() else Path(os.path.abspath(candidate))

    def subprocess_env(self) -> dict[str, str]:
        """Environment for the forecast_engine subprocess LocalRunner
        launches: everything this backend process already has, plus the
        Azure OpenAI / MLflow values this `Settings` object loaded from
        `.env` — explicitly forwarded, since a child process only inherits
        real OS environment variables, never pydantic-settings' parsed
        values. This is what makes editing `backend/.env` alone enough to
        configure forecast_engine, with no separate `.env` file needed
        inside forecast_engine/ itself.

        A field left `None` (unset) is simply omitted, never forwarded as
        the literal string "None" — forecast_engine's own config classes
        already treat an absent variable as "not configured" correctly.
        """
        forwarded = {
            "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
            "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
            "AZURE_OPENAI_DEPLOYMENT_NAME": self.azure_openai_deployment_name,
            "AZURE_OPENAI_API_VERSION": self.azure_openai_api_version,
            "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
            "MLFLOW_REGISTRY_URI": self.mlflow_registry_uri,
            "MLFLOW_EXPERIMENT_NAME": self.mlflow_experiment_name,
            "MLFLOW_ARTIFACT_LOCATION": self.mlflow_artifact_location,
        }
        env = os.environ.copy()
        env.update({key: value for key, value in forwarded.items() if value is not None})
        return env


@lru_cache
def get_settings() -> Settings:
    return Settings()
