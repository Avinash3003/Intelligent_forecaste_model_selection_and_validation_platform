"""Derived (engineered) feature selection — Priority C.

The single authoritative list of the derived feature columns a user can
choose to include, and how a chosen selection resolves into the model
parameters `SupervisedTreeModel` (`s05_models/base_model.py`) already reads
(`lags`, `rolling_windows`, `calendar_features`) — no parallel
feature-engineering system, this is the existing one, made user-selectable.

Scope, deliberately: only the lag / rolling-mean / calendar features the
gradient-boosted tree models (XGBoost, LightGBM) already generate
internally for themselves are exposed here. Every other model family
(Prophet, ARIMA, TFT, seasonal_naive) has its own native handling of trend
and seasonality and never reads `lags`/`rolling_windows`/`calendar_features`
at all — see `SupervisedTreeModel`'s own docstring. Nothing here is
written into the shared curated dataset every model trains from: these
features are computed per-model, at train *and* predict time, from the
same `_feature_row` builder, which is what keeps them leak-free (a lag or
rolling mean is only ever built from values strictly before the row it
describes) and valid for forward forecasting (nothing here depends on a
value the model would not also have available at inference time).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forecast_engine.config.model_config import ModelConfig

# The feature IDs a user may select — also the exact column name each one
# produces, so a selection reads directly as the resulting curated-preview
# column list (Step 5). Matches `SupervisedTreeModel.DEFAULT_LAGS` /
# `DEFAULT_ROLLING_WINDOWS` / `_CALENDAR_FEATURE_NAMES` exactly; nothing
# here is invented independently of that class.
SUPPORTED_LAGS: tuple[int, ...] = (1, 2, 3, 12)
SUPPORTED_ROLLING_WINDOWS: tuple[int, ...] = (3, 6)
SUPPORTED_CALENDAR_FEATURES: tuple[str, ...] = ("month", "quarter")


@dataclass(frozen=True)
class DerivedFeatureOption:
    """One selectable feature: its id (== resulting column name) and a
    short label for display."""

    feature_id: str
    label: str


def _lag_options() -> tuple[DerivedFeatureOption, ...]:
    return tuple(DerivedFeatureOption(f"lag_{n}", f"Lag {n}") for n in SUPPORTED_LAGS)


def _rolling_options() -> tuple[DerivedFeatureOption, ...]:
    return tuple(
        DerivedFeatureOption(f"rolling_mean_{n}", f"Rolling Mean ({n})") for n in SUPPORTED_ROLLING_WINDOWS
    )


def _calendar_options() -> tuple[DerivedFeatureOption, ...]:
    return tuple(DerivedFeatureOption(name, name.capitalize()) for name in SUPPORTED_CALENDAR_FEATURES)


# Every selectable feature, in the order the UI presents them.
DERIVED_FEATURE_OPTIONS: tuple[DerivedFeatureOption, ...] = _lag_options() + _rolling_options() + _calendar_options()

SUPPORTED_FEATURE_IDS: frozenset[str] = frozenset(option.feature_id for option in DERIVED_FEATURE_OPTIONS)

# The selection that reproduces today's behavior exactly — every feature
# `SupervisedTreeModel.DEFAULT_LAGS`/`DEFAULT_ROLLING_WINDOWS`/its calendar
# default already generates unprompted. A run that never mentions
# `derived_features` at all (every run before this feature existed) is
# `resolve_derived_feature_params(None)`, which returns exactly this.
DEFAULT_SELECTED_FEATURE_IDS: frozenset[str] = SUPPORTED_FEATURE_IDS


def validate_feature_ids(requested: list[str]) -> tuple[list[str], list[str]]:
    """Split a requested feature-id list into (valid, rejected).

    Never trusts the caller: an unsupported name is reported back, never
    silently substituted or allowed through.
    """
    valid = [feature_id for feature_id in requested if feature_id in SUPPORTED_FEATURE_IDS]
    rejected = [feature_id for feature_id in requested if feature_id not in SUPPORTED_FEATURE_IDS]
    return valid, rejected


def resolve_derived_feature_params(selected: list[str] | None) -> dict[str, object]:
    """Turn a selected feature-id list into the `lags` / `rolling_windows`
    / `calendar_features` params `SupervisedTreeModel` already reads.

    `None` (never selected — every run before this feature existed, or a
    caller that simply omits the field) resolves to exactly today's
    default behavior, unchanged. Unsupported ids are dropped rather than
    raised here — the API layer is where an unsupported name is rejected
    with a clear error; by the time a selection reaches the engine it has
    already been validated, and the engine stays defensive rather than
    trusting that.
    """
    if selected is None:
        selected_ids = DEFAULT_SELECTED_FEATURE_IDS
    else:
        valid, _rejected = validate_feature_ids(selected)
        selected_ids = frozenset(valid)

    lags = [n for n in SUPPORTED_LAGS if f"lag_{n}" in selected_ids]
    rolling_windows = [n for n in SUPPORTED_ROLLING_WINDOWS if f"rolling_mean_{n}" in selected_ids]
    calendar_features = [name for name in SUPPORTED_CALENDAR_FEATURES if name in selected_ids]

    return {
        "lags": lags,
        "rolling_windows": rolling_windows,
        "calendar_features": calendar_features,
    }


# Model names whose adapter actually reads `lags`/`rolling_windows`/
# `calendar_features` — see `SupervisedTreeModel` (s05_models/base_model.py).
# Every other registered model (Prophet, ARIMA, TFT, seasonal_naive) has its
# own native trend/seasonality handling and would simply ignore these keys,
# so applying a selection to their specs would be a no-op at best and
# confusing configuration at worst — this list is what keeps the selection
# scoped to the models it is actually valid for.
_DERIVED_FEATURE_AWARE_MODELS = frozenset({"xgboost", "lightgbm"})


def apply_to_model_config(model_config: ModelConfig, selected: list[str] | None) -> ModelConfig:
    """Return a new `ModelConfig` with the resolved derived-feature params
    merged into the tree-based models' `default_params`, every other model
    spec untouched.

    Applied once, before any collaborator (trainer, evaluator, SHAP engine,
    production selector) is constructed from `ModelConfig` — each of those
    builds its own `ModelRegistry` snapshot at construction time, so this
    must happen upstream of all of them rather than mutating specs after
    the fact. `selected=None` (a run that never mentions derived features)
    returns `model_config` completely unchanged — no new ModelConfig
    object, no behavior change, for every run before this feature existed.
    """
    if selected is None:
        return model_config

    resolved_params = resolve_derived_feature_params(selected)
    new_registry = tuple(
        replace(spec, default_params={**spec.default_params, **resolved_params})
        if spec.name.lower() in _DERIVED_FEATURE_AWARE_MODELS
        else spec
        for spec in model_config.registry
    )
    return replace(model_config, registry=new_registry)
