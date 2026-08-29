import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration, read from .env or real environment variables.

    Nothing is hardcoded: defaults are local-friendly and every deployment
    overrides what it needs.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Forecast IQ API"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    debug: bool = True

    cors_origins: list[str] = ["http://localhost:5173"]

    upload_dir: str = "uploads"
    max_upload_size_mb: int = 200

    # Where runs execute: "local" (subprocess) or "databricks" (a Databricks
    # Jobs API run — existing compute or a job cluster this backend creates).
    # Changing environments is this one value; an unknown value fails loudly
    # at startup.
    execution_mode: str = "local"

    # Where forecast_engine and its venv live, relative to the working
    # directory — never absolute, so any checkout works.
    forecast_engine_root: str = "../forecast_engine"

    # Override the engine interpreter; derived from the root when unset.
    forecast_engine_python: str | None = None

    # Override the LLM evaluation report path; derived from the root when unset.
    llm_eval_report_path: str | None = None

    job_poll_interval_seconds: float = 2.0
    job_timeout_seconds: float = 3600.0

    # The engine reads these from its own process environment (it runs as a
    # subprocess, not as imported code). Declaring them here keeps one .env;
    # subprocess_env() below is what actually forwards them.
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None

    # USD per 1,000 tokens. Unset means cost is reported as unavailable
    # rather than guessed.
    azure_openai_price_input_per_1k: float | None = None
    azure_openai_price_output_per_1k: float | None = None
    llm_max_tokens_per_run: int | None = None

    mlflow_tracking_uri: str | None = None
    mlflow_registry_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_artifact_location: str | None = None

    # --- Entra ID authentication ---
    # Off by default so a fresh checkout runs with no Azure tenant. main.py
    # refuses to start with it off outside development.
    auth_enabled: bool = False
    # Role granted to the local development identity when auth is off.
    dev_identity_role: str = "Admin"

    entra_tenant_id: str | None = None
    # This API's client id. Also accepted as a token audience.
    entra_api_client_id: str | None = None
    # Audience the frontend requests tokens for, normally api://<client id>.
    entra_api_audience: str | None = None
    # Global cloud; sovereign clouds use a different host.
    entra_authority_host: str = "https://login.microsoftonline.com"
    # Fallback for tenants using security groups instead of app roles:
    # JSON of {"<group object id>": "<Role>"}.
    entra_group_role_map: str | None = None

    # Public identifier the frontend needs to sign in. No secret is served here.
    entra_spa_client_id: str | None = None

    # --- Databricks execution ---
    # A service principal is preferred over a PAT: workspace-scoped,
    # rotatable, and not tied to a person.
    databricks_host: str | None = None
    databricks_client_id: str | None = None
    databricks_client_secret: str | None = None
    # Legacy alternative for workspaces that only issue PATs.
    databricks_token: str | None = None
    databricks_workspace_id: str | None = None

    # All-purpose cluster offered in the UI as the "existing compute"
    # fallback. Empty simply hides that option; nothing here is required.
    databricks_existing_cluster_id: str | None = None

    # Engine wheel a user-configured job cluster installs at run time.
    # Empty relies on the cluster already having it.
    databricks_engine_wheel_path: str | None = None

    # Databricks Container Services: the pre-built runtime image a NEW job
    # cluster pulls instead of resolving its dependencies from the
    # runtime's own environment. Empty disables DCS entirely — a new job
    # cluster is then built exactly as it was before this setting existed
    # (an ML runtime, no docker_image), the same "blank means off" idiom
    # every other optional Databricks feature on this class already uses.
    #
    # The existing all-purpose cluster is never affected by this setting;
    # attaching an image to it is a manual, one-time step the operator
    # performs in the Databricks UI (see docs), not something this backend
    # does automatically.
    # Where a Container Services run stages its inputs and outputs.
    #
    # A DCS run cannot use the UC Volume roots below. Replacing the runtime
    # image removes Databricks' own `uc-volumes` storage-scheme handler, so
    # the container's DBFS client cannot resolve /Volumes at all — proven on
    # a real DCS cluster, which reports `Unrecognized storage scheme:
    # uc-volumes` in /dbfs/Volumes/mount.err while Spark itself still has
    # Unity Catalog fully enabled. That is a property of DCS, not of this
    # workspace or of the image we build, and it applies to any cluster
    # carrying a docker_image.
    #
    # Workspace files are reachable from inside the container (verified: a
    # read/write round-trip under /Workspace/Shared succeeds), so a DCS run
    # stages here instead. Configurable rather than derived, so a deployment
    # can point it at a folder its principal owns.
    databricks_workspace_staging_root: str = "/Workspace/Shared/forecastiq/runs"

    databricks_docker_image_url: str | None = None
    # Basic auth Databricks presents to the private ACR repository at pull
    # time. Both blank is the common case (a public or already-cached
    # image); set together, never one without the other.
    databricks_docker_image_username: str | None = None
    databricks_docker_image_password: str | None = None

    # UC volume over the ADLS uploads container: staged datasets and run output.
    databricks_volumes_root: str = "/Volumes/forecastiq/forecasting/forecast_files"

    # UC volume for the curated dataset. Separate from uploads so raw and
    # derived data keep their own lifecycle. Cloud execution only.
    databricks_curated_volumes_root: str = "/Volumes/forecastiq/forecasting/curated_files"

    # UC volume for each key's winning fitted model.
    databricks_models_volumes_root: str = "/Volumes/forecastiq/forecasting/models_files"

    # UC volume for the exported forecast CSV.
    databricks_forecasts_volumes_root: str = "/Volumes/forecastiq/forecasting/forecasts_files"

    # UC volume for a blob-accessible copy of insights and the LLM trace.
    databricks_artifacts_volumes_root: str = "/Volumes/forecastiq/forecasting/artifacts_files"

    azure_storage_connection_string: str | None = None
    # Dataset preview reads uploads back from ADLS. A read-only, expiring
    # container SAS is preferred so the backend never holds account-wide rights.
    azure_storage_account: str | None = None
    azure_storage_sas_token: str | None = None
    azure_uploads_container: str = "uploads"

    # --- Cost estimation ---
    # Blended hourly compute rate (VM + DBU). Unset by default because the
    # right figure is subscription-specific; unset hides cost rather than
    # showing a made-up number.
    compute_cost_per_hour: float | None = None
    compute_cost_currency: str = "USD"

    azure_tenant_id: str | None = None
    azure_client_id: str | None = None
    azure_client_secret: str | None = None

    @property
    def entra_group_role_map_parsed(self) -> dict[str, str]:
        """`entra_group_role_map` as a dict, or empty if unset/malformed.

        Deliberately fails soft: a broken mapping must not take the API
        down, and an empty map simply means no group grants any role — a
        closed door, which is the safe direction for an authorization
        setting to fail in.
        """
        raw = (self.entra_group_role_map or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {str(key): str(value) for key, value in parsed.items()}

    @property
    def is_production_like(self) -> bool:
        """Anything that is not local development — gates the auth startup check."""
        return (self.app_env or "").strip().lower() not in {"development", "dev", "local", "test"}

    @property
    def upload_path(self) -> Path:
        # Absolute, because the staged path is handed to the engine
        # subprocess, which runs from a different working directory.
        path = Path(self.upload_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def engine_working_dir(self) -> Path:
        """Working directory of the engine subprocess: forecast_engine's parent.

        Every relative path the engine writes (notably its local MLflow store)
        lands here, so this backend must resolve those same paths against it.
        """
        return Path(self.forecast_engine_root).resolve().parent

    @property
    def llm_eval_report_path_resolved(self) -> Path:
        """Where the LLM evaluation CLI writes its report and the API reads it.

        Not run-scoped: the regression suite is a standalone, on-demand check.
        """
        if self.llm_eval_report_path:
            return Path(self.llm_eval_report_path)
        return Path(self.forecast_engine_root).resolve() / "s11_llm" / "eval_output" / "latest_regression_report.json"

    @property
    def mlflow_tracking_uri_resolved(self) -> str:
        """The tracking store this backend reads, matching what the engine writes.

        Configured values are used as-is. Unset falls back per execution
        mode, because the two modes write to different stores and there is
        no single unset default that matches both:

        - local: the engine subprocess runs on this same box with no
          explicit MLFLOW_TRACKING_URI, so it falls through to its own
          sqlite default (mlflow.db next to forecast_engine/). Matching
          that path is what lets both processes open the same file.
        - databricks: the engine runs as a Databricks job,
          where MLflow's own unset-default resolves to the workspace's
          managed tracking store — never the sqlite file above, which that
          job never touches. A deployment that sets DATABRICKS_HOST/credentials for
          job submission but never separately sets MLFLOW_TRACKING_URI
          previously fell through to the sqlite branch here, silently
          reading a store the engine never wrote to: run history and
          Experiments/Observability would show only whatever this
          backend's own process still held in memory, and nothing from
          before its last restart — exactly the "runs vanish" symptom this
          mirrors "databricks" to fix.
        """
        configured = (self.mlflow_tracking_uri or "").strip()
        if configured:
            return configured
        if (self.execution_mode or "").strip().lower() == "databricks":
            return "databricks"
        return f"sqlite:///{self.engine_working_dir / 'mlflow.db'}"

    @property
    def mlflow_experiment_name_resolved(self) -> str:
        # Mirrors the engine's own default so an unconfigured deployment
        # still reads the experiment the engine wrote to.
        return (self.mlflow_experiment_name or "").strip() or "/forecast-engine"

    @property
    def forecast_engine_python_path(self) -> Path:
        """The interpreter LocalRunner runs the engine with.

        Defaults to the engine's own venv, since it carries heavy deps
        (xgboost, statsmodels, mlflow) best kept out of this process.

        Always absolute, because the subprocess runs with cwd set to the
        engine directory. Absolutized with abspath rather than resolve():
        .venv/bin/python is a symlink, and following it to the base
        interpreter would lose venv activation and its dependencies.
        """
        if self.forecast_engine_python:
            candidate = Path(self.forecast_engine_python)
        else:
            candidate = Path(self.forecast_engine_root) / ".venv" / "bin" / "python"
        return candidate if candidate.is_absolute() else Path(os.path.abspath(candidate))

    def subprocess_env(self) -> dict[str, str]:
        """Environment for the engine subprocess.

        This process's own environment plus the Azure OpenAI / MLflow values
        loaded from .env — forwarded explicitly, because a child inherits
        real OS variables but not pydantic-settings' parsed ones. Unset
        fields are omitted rather than forwarded as the string "None".
        """
        forwarded = {
            "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
            "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
            "AZURE_OPENAI_DEPLOYMENT_NAME": self.azure_openai_deployment_name,
            "AZURE_OPENAI_API_VERSION": self.azure_openai_api_version,
            # Popen's env requires strings; these three are the numeric ones.
            "AZURE_OPENAI_PRICE_INPUT_PER_1K": _stringify(self.azure_openai_price_input_per_1k),
            "AZURE_OPENAI_PRICE_OUTPUT_PER_1K": _stringify(self.azure_openai_price_output_per_1k),
            "LLM_MAX_TOKENS_PER_RUN": _stringify(self.llm_max_tokens_per_run),
            "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
            "MLFLOW_REGISTRY_URI": self.mlflow_registry_uri,
            "MLFLOW_EXPERIMENT_NAME": self.mlflow_experiment_name,
            "MLFLOW_ARTIFACT_LOCATION": self.mlflow_artifact_location,
        }
        env = os.environ.copy()
        env.update({key: value for key, value in forwarded.items() if value is not None})
        return env


# Stringify for a subprocess env, or None to omit. Popen rejects non-strings.
def _stringify(value: float | int | None) -> str | None:
    return None if value is None else str(value)


@lru_cache
def get_settings() -> Settings:
    return Settings()
