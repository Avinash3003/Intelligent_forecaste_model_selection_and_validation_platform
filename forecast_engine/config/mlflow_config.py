"""MLflow Experiment Tracking & Model Registry configuration (Section 6.13).

The only thing that should ever change between Local Development and
Databricks is the *tracking URI* — every other MLflow behaviour (what gets
logged, when, and how) is identical in both environments. Centralizing
every MLflow-facing setting here, read from environment variables, is what
makes that true: `forecast_engine/tracking/` never hardcodes a path or a
URI, so pointing it at Databricks later is a deployment change, not a code
change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Standard MLflow environment variable names. `MLFLOW_TRACKING_URI` and
# `MLFLOW_REGISTRY_URI` are MLflow's own conventional variable names — the
# SDK itself reads them, so using the same names here means a deployment's
# existing MLflow environment configuration works without translation.
MLFLOW_TRACKING_URI_ENV_VAR = "MLFLOW_TRACKING_URI"
MLFLOW_REGISTRY_URI_ENV_VAR = "MLFLOW_REGISTRY_URI"
MLFLOW_EXPERIMENT_NAME_ENV_VAR = "MLFLOW_EXPERIMENT_NAME"
MLFLOW_ARTIFACT_LOCATION_ENV_VAR = "MLFLOW_ARTIFACT_LOCATION"

_DEFAULT_EXPERIMENT_NAME = "/forecast-engine"

# MLflow's own bare local file store ("./mlruns" with no URI scheme) is in
# maintenance mode as of the current SDK and refuses new experiments; the
# documented, recommended local backend is now a SQLite database for
# tracking *metadata*, with artifact *files* still landing in a local
# "mlruns/" folder — exactly Local Development's requirement, just split
# across the two URIs MLflow itself distinguishes.
_DEFAULT_LOCAL_TRACKING_URI = "sqlite:///mlflow.db"
_DEFAULT_LOCAL_ARTIFACT_LOCATION = "./mlruns"


@dataclass(frozen=True)
class MLflowConfig:
    """Root configuration for the MLflow tracking & registry layer."""

    enabled: bool = True

    # Defaults to a local SQLite store. Set to a Databricks URI
    # ("databricks", or a workspace profile) to track there instead — the
    # only change required to retarget environments.
    tracking_uri: str = _DEFAULT_LOCAL_TRACKING_URI

    # Model Registry URI, defaults to the tracking URI when unset (MLflow's
    # own behaviour). Declared for future use — Unity Catalog registration
    # is a URI value ("databricks-uc"), not a code change, once enabled.
    registry_uri: str | None = None

    experiment_name: str = _DEFAULT_EXPERIMENT_NAME

    # Where MLflow stores artifact files for experiments it creates.
    # Defaults to a local "mlruns/" folder; set to a cloud URI (e.g.
    # "abfss://...", "dbfs:/...") in a deployment that needs one.
    artifact_location: str = _DEFAULT_LOCAL_ARTIFACT_LOCATION

    # Prefix for registered model names — one registered model per
    # forecasting group is named "{prefix}-{sanitized_group_id}", which
    # reads naturally as a Unity Catalog three-level name's final segment
    # once a catalog/schema prefix is added ahead of it later.
    registered_model_name_prefix: str = "forecast_engine"

    log_artifacts: bool = True
    register_winner_model: bool = True

    # A run spanning thousands of forecasting groups would otherwise log an
    # unbounded number of MLflow parameters; the complete, unbounded set of
    # hyperparameters is always available as a JSON artifact regardless of
    # this cap.
    max_hyperparameter_params: int = 200

    # Per-key figures (forecast, model comparison, drift overlay) are drawn per
    # forecasting group, so their cost grows with the dataset: 10 groups is 30
    # figures, 500 groups would be 1,500. They are browsing aids in the MLflow
    # UI — nothing in the platform reads them, which reads only
    # `run/summary.json` — so a sample conveys the same thing as the full set
    # at a fraction of the time. Run-level summary charts are unaffected: there
    # are three of them regardless of dataset size.
    max_plot_groups: int = 10

    # Return the standard configuration: credentials from the environment
    @classmethod
    def default(cls) -> "MLflowConfig":
        return cls.from_env()

    # Build configuration from the standard MLflow environment variables
    @classmethod
    def from_env(cls) -> "MLflowConfig":
        # An unset MLFLOW_TRACKING_URI is not an error — it means "track
        # locally" against the local SQLite default, exactly as an unset
        # variable means "track to Databricks" once a deployment sets it.
        return cls(
            tracking_uri=_env(MLFLOW_TRACKING_URI_ENV_VAR, _DEFAULT_LOCAL_TRACKING_URI),
            registry_uri=_env(MLFLOW_REGISTRY_URI_ENV_VAR),
            experiment_name=_env(MLFLOW_EXPERIMENT_NAME_ENV_VAR, _DEFAULT_EXPERIMENT_NAME),
            artifact_location=_env(MLFLOW_ARTIFACT_LOCATION_ENV_VAR, _DEFAULT_LOCAL_ARTIFACT_LOCATION),
        )

    # Build from a flat mapping, ignoring unknown keys
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MLflowConfig":
        if not payload:
            return cls()
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in allowed})


# Read an environment variable, treating blank as unset
def _env(name: str, default: str | None = None) -> str | None:
    # A caller that exports MLFLOW_TRACKING_URI= (as a .env file listing
    # optional settings does) means "not configured", not "configure me with
    # an empty string" — without this it would silently override the local
    # defaults and break tracking.
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default
