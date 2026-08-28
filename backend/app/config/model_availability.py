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

# execution_mode -> {model id: why it cannot run there}. A mode absent from
# this table can run everything the engine registers, which is the correct
# default: a new mode is assumed capable until something is known to be
# missing from it.
_UNAVAILABLE_BY_MODE: dict[str, dict[str, str]] = {}

# Models offered in the picker but NOT actually executed.
#
# TFT needs torch and pytorch-forecasting (~900 MB), which today's cloud
# compute does not carry unless a Container Services image supplies them —
# so on a plain runtime it genuinely cannot run. It is nonetheless left
# selectable, by product decision,
# because the picker is part of the demo narrative; it is then dropped from
# the model list before the run is submitted (see
# deployment_service.build_execution_request).
#
# The consequence, stated plainly so nobody debugs it as a fault: selecting
# TFT changes nothing about a run. It will not appear in the results, the
# comparison table or MLflow, because it was never trained. Removing this
# entry is all it takes to make the selection real again once the runtime
# carries torch.
SILENTLY_SKIPPED_MODELS: frozenset[str] = frozenset({"tft"})


def strip_silently_skipped(model_ids: list[str] | None) -> list[str] | None:
    """`model_ids` minus anything in SILENTLY_SKIPPED_MODELS, order kept.

    Returns None unchanged so "no explicit selection" keeps meaning "the
    engine's own defaults", rather than becoming an empty list that would
    train nothing at all.
    """
    if model_ids is None:
        return None
    return [m for m in model_ids if m.strip().lower() not in SILENTLY_SKIPPED_MODELS]


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
