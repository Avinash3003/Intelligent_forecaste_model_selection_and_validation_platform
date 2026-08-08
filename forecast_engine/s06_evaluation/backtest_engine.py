"""Rolling / expanding backtesting (Section 6.4).

Scores a model by replaying history: fit on data up to a point, forecast
the window that follows, compare against what actually happened, then
advance and repeat. Two window strategies are supported and chosen by
configuration — ROLLING keeps a fixed-length training window, EXPANDING
grows it from a fixed origin.

Three properties matter:

  * Group isolation. Every fold is fit on one forecasting group's series
    alone. Nothing is pooled across business keys, so one key's behaviour
    can never leak into another's score.
  * Honest refitting. Each fold constructs a *fresh* model from the
    registry using the parameters Phase 6 selected, and fits it only on
    that fold's training window. Reusing the already-fitted model would
    score it on data it had already seen.
  * No ranking. This produces evidence, not a verdict — Section 6.4 is
    explicit that the absolute accuracy value is one structured input to
    the later ranking stage, never the selection criterion itself.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from typing import Any

import pandas as pd

from forecast_engine.config.evaluation_config import BacktestConfig, BacktestStrategy, MetricsConfig
from forecast_engine.s06_evaluation.evaluation_report import BacktestResult, BacktestWindowResult
from forecast_engine.s06_evaluation.metrics import ForecastMetrics, MetricsCalculator
from forecast_engine.s05_models.base_model import TrainedModel
from forecast_engine.s05_models.model_registry import ModelRegistry
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries


class BacktestEngine:
    """Runs rolling or expanding backtests for one group/model pair."""

    # Wire up the model registry and backtest/metrics configuration
    def __init__(
        self,
        registry: ModelRegistry,
        config: BacktestConfig | None = None,
        metrics_config: MetricsConfig | None = None,
    ) -> None:
        self._registry = registry
        self._config = config or BacktestConfig()
        self._metrics_config = metrics_config or MetricsConfig()
        self._calculator = MetricsCalculator(self._metrics_config)

    # Backtest one model against one group's series
    def run(self, trained: TrainedModel, series: ForecastSeries) -> BacktestResult:
        result = BacktestResult(strategy=self._config.strategy.value)

        windows = self._build_windows(series.observation_count)
        if not windows:
            result.skipped_reason = (
                f"Series has {series.observation_count} observation(s); at least "
                f"{self._config.min_train_size + self._config.horizon} are required to backtest "
                f"with a {self._config.horizon}-period horizon."
            )
            return result

        spec = self._registry.config.find(trained.model_name)
        if spec is None:
            result.skipped_reason = f"Model '{trained.model_name}' is no longer registered."
            return result

        # Accumulated across folds so the overall metrics reflect every
        # scored observation, not an average of averages.
        all_actuals: list[float] = []
        all_predictions: list[float] = []
        horizon_actuals: dict[int, list[float]] = {}
        horizon_predictions: dict[int, list[float]] = {}

        for index, (train_start, train_end) in enumerate(windows):
            window = self._evaluate_window(
                spec_name=trained.model_name,
                params=trained.params,
                series=series,
                train_start=train_start,
                train_end=train_end,
                index=index,
            )
            if window is None:
                continue

            window_result, actuals, predictions = window
            result.windows.append(window_result)
            all_actuals.extend(actuals)
            all_predictions.extend(predictions)

            # Step 1 is the first period beyond the training window, so
            # accuracy decay with distance stays visible.
            for step, (actual, prediction) in enumerate(zip(actuals, predictions), start=1):
                horizon_actuals.setdefault(step, []).append(actual)
                horizon_predictions.setdefault(step, []).append(prediction)

        if not all_actuals:
            result.skipped_reason = "No backtest window could be evaluated for this model."
            return result

        result.overall = self._calculator.calculate(all_actuals, all_predictions)

        if self._metrics_config.compute_per_horizon:
            result.per_horizon = {
                step: self._calculator.calculate(horizon_actuals[step], horizon_predictions[step])
                for step in sorted(horizon_actuals)
            }

        return result

    # Compute (train_start, train_end) index pairs for each fold
    def _build_windows(self, observation_count: int) -> list[tuple[int, int]]:
        horizon = self._config.horizon
        min_train = self._config.min_train_size
        step = self._config.step or horizon

        if observation_count < min_train + horizon:
            return []

        windows: list[tuple[int, int]] = []
        train_end = min_train

        while train_end + horizon <= observation_count and len(windows) < self._config.max_windows:
            if self._config.strategy is BacktestStrategy.EXPANDING:
                # Origin stays fixed; the window grows with each fold.
                train_start = 0
            else:
                # Fixed-length window that slides forward.
                size = self._config.rolling_window_size or min_train
                train_start = max(0, train_end - size)

            windows.append((train_start, train_end))
            train_end += step

        return windows

    # Fit on one training window and score the window that follows
    def _evaluate_window(
        self,
        spec_name: str,
        params: dict[str, Any],
        series: ForecastSeries,
        train_start: int,
        train_end: int,
        index: int,
    ) -> tuple[BacktestWindowResult, list[float], list[float]] | None:
        horizon = self._config.horizon
        frame = series.frame

        train_frame = frame.iloc[train_start:train_end]
        test_frame = frame.iloc[train_end : train_end + horizon]

        if train_frame.empty or test_frame.empty:
            return None

        # A view of the same series restricted to the training window, so
        # the adapter sees exactly the interface it expects.
        train_series = dataclass_replace(series, frame=train_frame.reset_index(drop=True))

        spec = self._registry.config.find(spec_name)
        if spec is None:
            return None

        try:
            model = self._registry.create(spec, params)
            model.initialize()
            model.train(train_series)

            # The test window is historical, so its regressor values are
            # genuinely available — unlike a true forward forecast.
            output = model.predict(horizon, future_frame=test_frame)
        except Exception:  # noqa: BLE001 - one unusable fold must not end the backtest
            return None

        actuals = [float(value) for value in test_frame[series.target_column].to_numpy()]
        predictions = [float(value) for value in output.values[: len(actuals)]]

        if len(predictions) != len(actuals):
            return None

        metrics: ForecastMetrics = self._calculator.calculate(actuals, predictions)

        window_result = BacktestWindowResult(
            window_index=index,
            train_start=_iso(train_frame[series.date_column].iloc[0]),
            train_end=_iso(train_frame[series.date_column].iloc[-1]),
            test_start=_iso(test_frame[series.date_column].iloc[0]),
            test_end=_iso(test_frame[series.date_column].iloc[-1]),
            train_size=int(len(train_frame)),
            metrics=metrics,
        )
        return window_result, actuals, predictions


# Render a timestamp as an ISO string for the report
def _iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()
