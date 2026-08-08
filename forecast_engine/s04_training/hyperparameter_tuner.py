"""Hyperparameter tuning (Section 6.3.2).

Selects parameters for models whose family benefits from a search — today
the gradient-boosted trees. Statistical families opt out via
`supports_tuning()` and select internally by their own criterion (ARIMA by
AIC), which is both cheaper and more appropriate than a cross-validated
search.

Two boundaries are worth being explicit about:

  * Search strategy is configuration, not code. GRID and RANDOM map to
    scikit-learn's searches; BAYESIAN and SUCCESSIVE_HALVING are declared
    for backends a later phase adds and fall back to RANDOM today rather
    than silently skipping tuning.
  * Scoring uses chronological splits (TimeSeriesSplit) so a candidate is
    never scored on data preceding its own training window. This is
    parameter selection *inside* model fitting — it is not the platform's
    Backtesting stage (Section 6.4), which evaluates finished models on
    rolling windows and is deliberately not implemented in this phase.
    When Backtesting lands, its harness replaces the splitter here so the
    search and the evaluation share one definition of a window.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from forecast_engine.config.model_config import SearchStrategy, TuningConfig
from forecast_engine.s05_models.base_model import BaseForecastingModel, SupervisedTreeModel
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries


class HyperparameterTuner:
    """Chooses model parameters via a configurable search."""

    # Store the tuning configuration
    def __init__(self, config: TuningConfig | None = None) -> None:
        self._config = config or TuningConfig()

    # Select parameters for a model on this group's series, never raising
    def tune(
        self, model: BaseForecastingModel, series: ForecastSeries
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        skip_reason = self._skip_reason(model, series)
        if skip_reason:
            return model.params, {"tuned": False, "reason": skip_reason}

        # Only the supervised (tree) families expose a design matrix that a
        # scikit-learn search can operate on.
        if not isinstance(model, SupervisedTreeModel):
            return model.params, {"tuned": False, "reason": "Model family does not support grid search."}

        try:
            return self._search(model, series)
        except Exception as exc:  # noqa: BLE001 - tuning must never lose a model
            return model.params, {"tuned": False, "reason": f"Search failed, using defaults: {exc}"}

    # Return why tuning should be skipped, or None to proceed
    def _skip_reason(self, model: BaseForecastingModel, series: ForecastSeries) -> str | None:
        if not self._config.enabled:
            return "Tuning disabled by configuration."
        if not model.supports_tuning():
            return "Model selects its parameters internally."
        if not model.spec.search_space:
            return "No search space declared for this model."
        if series.observation_count < self._config.min_observations_for_tuning:
            # Splitting a short series leaves folds too small to rank
            # candidates meaningfully; the defaults are the honest choice.
            return (
                f"Series has {series.observation_count} observation(s), below the "
                f"{self._config.min_observations_for_tuning} required to tune."
            )
        return None

    # Run the configured search and return the winning parameters
    def _search(
        self, model: SupervisedTreeModel, series: ForecastSeries
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

        features, target = model.build_design_matrix(series)

        # Guarantee every fold has data even on modestly sized series.
        splits = max(2, min(self._config.cv_splits, len(target) // 4))
        splitter = TimeSeriesSplit(n_splits=splits)

        estimator = model.build_estimator()
        strategy = model.spec.search_strategy
        space = model.spec.search_space

        if strategy is SearchStrategy.GRID:
            search = GridSearchCV(
                estimator,
                param_grid=space,
                cv=splitter,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
            )
        else:
            # RANDOM, and the not-yet-implemented BAYESIAN /
            # SUCCESSIVE_HALVING backends, all resolve to a randomized
            # search so a declared strategy never means "no tuning".
            search = RandomizedSearchCV(
                estimator,
                param_distributions=space,
                n_iter=model.spec.trial_budget,
                cv=splitter,
                scoring="neg_mean_absolute_error",
                random_state=self._config.random_state,
                n_jobs=1,
            )

        search.fit(features, target)

        tuned_params = {**model.params, **search.best_params_}
        metadata = {
            "tuned": True,
            "strategy": strategy.value,
            "cv_splits": splits,
            "best_params": dict(search.best_params_),
            "best_score_mae": abs(float(search.best_score_)),
            "candidates_evaluated": int(len(pd.DataFrame(search.cv_results_))),
        }
        return tuned_params, metadata
