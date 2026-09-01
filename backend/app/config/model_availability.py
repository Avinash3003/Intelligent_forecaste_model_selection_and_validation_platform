"""Which forecasting models the *current execution mode* can actually run.

A model being registered in the engine does not mean the environment it
will execute in has the library behind it. The clearest case is TFT: torch
and pytorch-forecasting are roughly 900 MB of dependencies that a cloud
run's compute does not carry unless something explicitly installs them —
today, that means a Databricks Container Services image built from
forecast_engine/requirements.txt (see the root Dockerfile); a plain ML
runtime with no such image attached does not have them either. Wherever
they are missing, `TemporalFusionTransformerModel.is_available()` returns
False and the engine reports the model Unavailable rather than training it.

Without this module the platform lies twice about that: the model picker
offers TFT as a choice, and the estimator charges its (high) per-fit weight
for work that will never happen. Both now read the same table.

This is availability *of the runtime*, not of the model — nothing here
changes the engine's own registry, its `min_observations` gates, or which
models it would train if the library were present.
"""

from __future__ import annotations

# The models a user can pick in the Configure step. Mirrors the engine's
# registry entries that are `enabled=True` (forecast_engine/config/
# model_config.py) — seasonal_naive is deliberately absent because it is the
# fallback, never a candidate models compete against.
CANDIDATE_MODEL_IDS: tuple[str, ...] = ("prophet", "arima", "lightgbm", "xgboost", "tft")

# Mirrors ModelConfig.DEFAULT_FALLBACK_MODEL (forecast_engine/config/
# model_config.py) -- verified against the engine's own source text by
# tests/backend/test_run_limits_match_the_engine.py, the same technique
# used for the horizon bounds in run_limits.py. A run with no explicit
# fallback_model falls back to exactly this on the engine side (see
# build_execution_request); reported here purely so the Configure step can
# pre-select the model a run will actually use, not a disconnected guess.
DEFAULT_FALLBACK_MODEL = "seasonal_naive"

# execution_mode -> {model id: why it cannot run there}. A mode absent from
# this table can run everything the engine registers, which is the correct
# default: a new mode is assumed capable until something is known to be
# missing from it.
_UNAVAILABLE_BY_MODE: dict[str, dict[str, str]] = {}

# Models that only run on the ForecastIQ container runtime.
#
# TFT needs torch and pytorch-forecasting (~900 MB). The DCS image installs
# and import-asserts both at build time, so a container run can execute it;
# a plain runtime cannot, and no amount of runtime pip installing should be
# used to paper over that.
#
# These used to be stripped from the model list silently, which meant
# selecting TFT changed nothing about a run and left no trace anywhere —
# not in the results, the comparison table or MLflow — with nothing to tell
# the user why. A request the platform cannot honour is now refused with a
# reason instead of being quietly discarded.
CONTAINER_ONLY_MODELS: frozenset[str] = frozenset({"tft"})


def unsupported_models(model_ids: list[str] | None, uses_container: bool) -> dict[str, str]:
    """The requested models this compute cannot run, mapped to why.

    Empty means the selection is executable as asked. Callers refuse the
    request on a non-empty result rather than trimming it: a model the user
    explicitly chose is not something to drop on their behalf.
    """
    if model_ids is None or uses_container:
        return {}
    return {
        model: (
            "This model needs the ForecastIQ container runtime, which supplies torch and "
            "pytorch-forecasting. Select New Job Compute to run it."
        )
        for model in model_ids
        if model.strip().lower() in CONTAINER_ONLY_MODELS
    }


def unavailable_models(execution_mode: str | None) -> dict[str, str]:
    """Model ids this execution mode cannot run, mapped to the reason."""
    mode = (execution_mode or "local").strip().lower()
    return dict(_UNAVAILABLE_BY_MODE.get(mode, {}))


def is_model_available(model_id: str, execution_mode: str | None) -> bool:
    return model_id.strip().lower() not in unavailable_models(execution_mode)


def filter_available(model_ids: list[str], execution_mode: str | None) -> list[str]:
    """`model_ids` minus anything this execution mode cannot run, order kept."""
    blocked = unavailable_models(execution_mode)
    return [model for model in model_ids if model.strip().lower() not in blocked]


def container_runtime_required(settings: object) -> bool:
    """Whether ForecastIQ only accepts container-backed execution."""
    return bool(getattr(settings, "databricks_require_container_runtime", False)) and bool(
        (getattr(settings, "databricks_docker_image_url", "") or "").strip()
    )


def compute_rejection_reason(compute: object, settings: object) -> str | None:
    """Why this compute selection cannot run a ForecastIQ pipeline, or None.

    The one place that decides. Callers refuse on a non-empty reason rather
    than quietly rerouting: a run that silently lands on a different runtime
    than the user chose is how models ended up available on one compute and
    missing on another.
    """
    if not container_runtime_required(settings):
        return None
    if getattr(compute, "mode", None) == "existing_compute":
        return (
            "ForecastIQ runs on its prebuilt container runtime, which carries every model "
            "dependency. Existing Compute cannot load that image, so it is retained only as "
            "legacy infrastructure. Select New Job Compute to run this pipeline."
        )
    return None
