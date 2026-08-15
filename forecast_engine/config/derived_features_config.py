"""The derived feature columns a user can select, and how they resolve.

Exposes only the lag / rolling-mean / calendar features the tree models
(XGBoost, LightGBM) already build for themselves — every other family
handles trend and seasonality natively and ignores these entirely.

Nothing here is written into the shared curated dataset: the features are
computed per model at both train and predict time from one builder, which
is what keeps them leak-free (a lag only ever uses values strictly before
its own row) and valid for forward forecasting.
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
    """Split requested ids into (valid, rejected) — an unsupported name is
    reported back, never silently allowed through."""
    valid = [feature_id for feature_id in requested if feature_id in SUPPORTED_FEATURE_IDS]
    rejected = [feature_id for feature_id in requested if feature_id not in SUPPORTED_FEATURE_IDS]
    return valid, rejected


def resolve_derived_feature_params(selected: list[str] | None) -> dict[str, object]:
    """Turn selected ids into the lags/rolling_windows/calendar_features the
    tree models already read.

    None means "not selected" and resolves to the existing defaults.
    Unsupported ids are dropped here rather than raised — the API rejects
    them with a clear error, and this stays defensive.
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
    """A new ModelConfig with these features merged into the tree models'
    defaults, every other spec untouched.

    Must run before any collaborator is built from the config, since each
    takes its own registry snapshot at construction. selected=None returns
    the original object unchanged.
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
