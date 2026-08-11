import json
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

    # Pipeline Orchestration & Execution Layer (Section 6.14). One of:
    #   local          runs forecast_engine directly, as a local subprocess.
    #   databricks     submits to the Databricks Serverless job (primary
    #                  cloud path — see app/orchestration/databricks_runner.py).
    #   databricks_dcs submits to the Databricks Container Services job (the
    #                  ACR/Docker path — see app/orchestration/dcs_runner.py).
    # Retargeting environments is this one value changing, never a code
    # change. An unrecognized value fails Runner construction with a clear
    # configuration error rather than silently falling back to another mode.
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

    # LLMOps (Section 13) — same names forecast_engine's own LLMConfig reads
    # (forecast_engine/config/llm_config.py), so this one .env is genuinely
    # the single place both the run and the pre-run estimate read pricing
    # and the token budget from. USD per 1,000 tokens; unset means the
    # estimate reports LLM cost as unavailable rather than a fabricated
    # figure — never a guessed default.
    azure_openai_price_input_per_1k: float | None = None
    azure_openai_price_output_per_1k: float | None = None
    llm_max_tokens_per_run: int | None = None

    mlflow_tracking_uri: str | None = None
    mlflow_registry_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_artifact_location: str | None = None

    # --- Entra ID authentication (Phase A) --------------------------------
    #
    # Off by default so a fresh checkout runs locally with no Azure tenant.
    # Every deployed environment sets AUTH_ENABLED=true; `main.py` refuses
    # to start with it false outside `development`, so "works on my laptop"
    # can never ship as "no authentication in production".
    auth_enabled: bool = False
    # Role granted to the local development identity when auth is off.
    dev_identity_role: str = "Admin"

    entra_tenant_id: str | None = None
    # App registration (client) id of *this API*. Also accepted as a token
    # audience, since Entra issues either this or the App ID URI depending
    # on how the registration is configured.
    entra_api_client_id: str | None = None
    # The audience the frontend requests tokens for — normally
    # `api://<entra_api_client_id>`.
    entra_api_audience: str | None = None
    # Sovereign/gov clouds use a different authority host; the default is
    # the global cloud.
    entra_authority_host: str = "https://login.microsoftonline.com"
    # Fallback when a tenant assigns access through security groups rather
    # than app roles: a JSON object of {"<group object id>": "<Role>"}.
    entra_group_role_map: str | None = None

    # Values the frontend needs to start a sign-in. All are public
    # identifiers by design — a SPA client id and authority are visible in
    # any browser — and no secret is ever served from here.
    entra_spa_client_id: str | None = None

    # --- Databricks execution (Phase A) -----------------------------------
    #
    # Read only when EXECUTION_MODE=databricks. Authentication prefers an
    # Entra service-principal (OAuth M2M) over a PAT: it is scoped to the
    # workspace, rotatable, and never a user's personal credential.
    databricks_host: str | None = None
    databricks_client_id: str | None = None
    databricks_client_secret: str | None = None
    # Legacy alternative to the service principal above. Supported because
    # some workspaces still only issue PATs; the service principal is
    # preferred wherever both are possible.
    databricks_token: str | None = None
    databricks_workspace_id: str | None = None

    # The existing bundle-deployed Job (databricks/resources/forecast_job.yml).
    # Resolved to a job id by name at submission time, so redeploying the
    # bundle — which mints a new job id — needs no configuration change here.
    databricks_job_name: str = "forecastiq-forecast-pipeline"
    # Set to pin an exact job id and skip the name lookup entirely.
    databricks_job_id: int | None = None

    # The DCS counterpart of the two settings above — read only when
    # EXECUTION_MODE=databricks_dcs. A separate job (and name/id), not a
    # second meaning for the same setting: Serverless and DCS are deployed
    # as two distinct Job resources (databricks/resources/
    # forecast_job_serverless.yml and forecast_job_dcs.yml) so redeploying
    # one bundle target never repoints the other mode at the wrong job.
    databricks_dcs_job_name: str = "forecastiq-forecast-pipeline-dcs"
    databricks_dcs_job_id: int | None = None

    # Unity Catalog external volume (over the ADLS `uploads` container) used
    # to stage datasets and read run output back. Matches
    # databricks.yml's `volumes_root`; no second storage location is created.
    databricks_volumes_root: str = "/Volumes/forecastiq/forecasting/forecast_files"

    # Unity Catalog external volume over the ADLS `curated` container, where
    # the engine writes the curated (post-preprocessing) dataset. A separate
    # volume from the one above so raw uploads and derived datasets keep
    # their own lifecycle and retention; both are existing containers, and
    # no new storage account is introduced.
    #
    # Read only for cloud execution. Local runs keep the engine's own
    # relative default, which is correct there because the filesystem
    # outlives the process.
    databricks_curated_volumes_root: str = "/Volumes/forecastiq/forecasting/curated_files"

    # Unity Catalog external volume over the ADLS `models` container,
    # where the engine writes each forecast key's winning fitted model.
    # Its own container so serialized models keep a lifecycle separate
    # from datasets; again an existing container, no new account.
    databricks_models_volumes_root: str = "/Volumes/forecastiq/forecasting/models_files"

    azure_storage_connection_string: str | None = None
    # Dataset preview reads the uploaded file back from ADLS. A container-
    # scoped read-only SAS is preferred over the connection string: it cannot
    # write and it expires, so the backend never holds account-wide rights.
    azure_storage_account: str | None = None
    azure_storage_sas_token: str | None = None
    azure_uploads_container: str = "uploads"

    # --- Cost estimation --------------------------------------------------
    #
    # Blended hourly rate for the compute a run occupies (VM + DBU). Left
    # unset by default: the correct figure depends on the subscription's
    # pricing agreement, and inventing one would put a fabricated number on
    # a screen people budget against. Unset simply hides cost from the
    # estimate and shows the duration alone.
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
        """Anything that is not explicitly local development.

        Used for the startup guard that forbids running with
        authentication disabled outside a developer's machine.
        """
        return (self.app_env or "").strip().lower() not in {"development", "dev", "local", "test"}

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
            # str()-converted: subprocess.Popen's env mapping requires
            # string values, and these three are the only numeric fields
            # forwarded here — every other one is already a string.
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


# str()-convert a value for a subprocess environment, or None to omit it —
# `subprocess.Popen`'s `env` mapping requires every value to be a string,
# unlike a plain dict, which silently accepts (and Popen then rejects) an
# int or float.
def _stringify(value: float | int | None) -> str | None:
    return None if value is None else str(value)


@lru_cache
def get_settings() -> Settings:
    return Settings()
