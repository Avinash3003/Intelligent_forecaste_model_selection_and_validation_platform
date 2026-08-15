"""Every candidate model, declared as data.

Which library backs it, its defaults, its search space and strategy, and its
minimum history. Nothing about a model's behaviour is hard-coded in the
training engine, so onboarding a new family is a config change.

default_params are what a model trains with when no search runs;
search_space is what a search may vary. Keeping them separate means
disabling tuning still leaves every model fully configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SearchStrategy(str, Enum):
    """How to search a model's parameter space.

    GRID for small enumerable spaces (ARIMA orders), RANDOM for larger ones
    where exhaustive search is wasteful (tree models), NONE to use the
    declared defaults as-is.

    BAYESIAN and SUCCESSIVE_HALVING are declared but fall back to RANDOM
    today, rather than silently skipping tuning.
    """

    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    SUCCESSIVE_HALVING = "successive_halving"
    NONE = "none"


@dataclass(frozen=True)
class ModelSpec:
    """One candidate model's declaration.

    adapter is resolved lazily, so an uninstalled library only affects the
    model that needs it. search_space is deliberately untyped — each family
    describes its own parameters and the engine just passes them on.
    min_observations is the least history the model can train on; a shorter
    group is skipped for this model, not failed.
    """

    name: str
    adapter: str
    enabled: bool = True
    default_params: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] = field(default_factory=dict)
    search_strategy: SearchStrategy = SearchStrategy.NONE
    trial_budget: int = 10
    supports_features: bool = True
    min_observations: int = 12


# The out-of-the-box candidate set named in Section 12.1.
DEFAULT_MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="prophet",
        adapter="forecast_engine.s05_models.prophet_model.ProphetModel",
        # Prophet has no cheap in-sample selection criterion, so it is
        # configuration-driven: the declared parameters are used directly.
        default_params={
            "seasonality_mode": "additive",
            "yearly_seasonality": "auto",
            "weekly_seasonality": "auto",
            "daily_seasonality": False,
            "changepoint_prior_scale": 0.05,
            "interval_width": 0.95,
        },
        search_strategy=SearchStrategy.NONE,
        supports_features=True,
        min_observations=24,
    ),
    ModelSpec(
        name="arima",
        adapter="forecast_engine.s05_models.arima_model.ArimaModel",
        default_params={"order": (1, 1, 1), "trend": None},
        # A small, enumerable space selected by AIC — the standard
        # information-criterion approach for statistical models.
        search_space={
            "p": [0, 1, 2],
            "d": [0, 1],
            "q": [0, 1, 2],
        },
        search_strategy=SearchStrategy.GRID,
        # ARIMA here is univariate; exogenous support is a later extension.
        supports_features=False,
        min_observations=20,
    ),
    ModelSpec(
        name="xgboost",
        adapter="forecast_engine.s05_models.xgboost_model.XGBoostModel",
        default_params={
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": 1,
        },
        search_space={
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
        },
        search_strategy=SearchStrategy.RANDOM,
        trial_budget=8,
        supports_features=True,
        min_observations=24,
    ),
    ModelSpec(
        name="lightgbm",
        adapter="forecast_engine.s05_models.lightgbm_model.LightGBMModel",
        default_params={
            "n_estimators": 300,
            "max_depth": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "n_jobs": 1,
            "verbose": -1,
        },
        search_space={
            "n_estimators": [200, 300, 500],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
        },
        search_strategy=SearchStrategy.RANDOM,
        trial_budget=8,
        supports_features=True,
        min_observations=24,
    ),
    ModelSpec(
        name="tft",
        adapter="forecast_engine.s05_models.tft_model.TemporalFusionTransformerModel",
        default_params={
            "max_epochs": 10,
            "hidden_size": 16,
            "attention_head_size": 2,
            "dropout": 0.1,
            "learning_rate": 0.03,
            "max_encoder_length": 24,
            "max_prediction_length": 12,
            "batch_size": 64,
        },
        search_space={
            "hidden_size": [8, 16, 32],
            "learning_rate": [0.01, 0.03, 0.1],
        },
        search_strategy=SearchStrategy.NONE,
        supports_features=True,
        # Deep models need substantially more history to be worth training.
        min_observations=60,
    ),
    ModelSpec(
        name="seasonal_naive",
        adapter="forecast_engine.s05_models.seasonal_naive_model.SeasonalNaiveModel",
        # Disabled by default: this is not a candidate models compete
        # against, so it never appears in the frontend's model picker or
        # trains during Phase 6. Final Production Model Selection (Section
        # 6.9) looks it up directly by name — ModelRegistry.find() and
        # .create() do not filter on `enabled`, so it remains reachable.
        enabled=False,
        default_params={"seasonal_period": 12},
        search_strategy=SearchStrategy.NONE,
        supports_features=False,
        min_observations=2,
    ),
)

# Fallback used when no candidate model survives validation (Section 6.9).
# Declared as config so a deployment can swap the baseline without a code
# change. Consumed by Phase 7B's Final Production Model Selection stage.
DEFAULT_FALLBACK_MODEL = "seasonal_naive"


@dataclass(frozen=True)
class TuningConfig:
    """Controls hyperparameter search behaviour across all models."""

    enabled: bool = True

    # Number of chronological splits used to score candidate parameter sets.
    # Time-ordered rather than random so a trial is never scored on data
    # that precedes its own training window.
    cv_splits: int = 3

    # Below this many observations a search is skipped entirely and the
    # declared defaults are used — splitting a short series leaves folds too
    # small to rank parameters meaningfully.
    min_observations_for_tuning: int = 48

    random_state: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """Root model configuration for a run."""

    registry: tuple[ModelSpec, ...] = DEFAULT_MODEL_REGISTRY
    fallback_model: str = DEFAULT_FALLBACK_MODEL
    tuning: TuningConfig = field(default_factory=TuningConfig)

    # Return the standard registry with every candidate enabled
    @classmethod
    def default(cls) -> "ModelConfig":
        return cls()

    # Return the models this run should train, filtered by user selection
    def enabled_models(self, selected: list[str] | None = None) -> tuple[ModelSpec, ...]:
        # Section 5.2 requires that no model outside the user's selection is
        # trained or considered, so every downstream stage must use this
        # rather than the raw registry.
        candidates = tuple(spec for spec in self.registry if spec.enabled)
        if selected is None:
            return candidates

        requested = {name.lower() for name in selected}
        return tuple(spec for spec in candidates if spec.name.lower() in requested)

    # Look up one model declaration by registry name
    def find(self, name: str) -> ModelSpec | None:
        for spec in self.registry:
            if spec.name.lower() == name.lower():
                return spec
        return None
