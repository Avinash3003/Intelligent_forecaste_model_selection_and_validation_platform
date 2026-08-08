"""Prophet adapter.

Prophet expects a two-column frame (`ds` timestamps, `y` values) and takes
exogenous regressors through `add_regressor` before fitting. Both are
handled here so no other stage needs to know Prophet's conventions.

Parameters are configuration-driven: Prophet exposes no cheap in-sample
selection criterion, so the registry's declared values are used directly
rather than searched.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

import pandas as pd

from forecast_engine.s05_models.base_model import BaseForecastingModel, ForecastOutput
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries

# Prophet's backend logs verbosely at INFO on every fit; quieted here so a
# multi-thousand-key run does not drown its own progress output.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


class ProphetModel(BaseForecastingModel):
    """Additive/multiplicative decomposition model via Prophet."""

    # Whether prophet is importable
    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("prophet") is not None

    # Build a fresh Prophet estimator from self.params
    def initialize(self) -> None:
        from prophet import Prophet

        # Regressors are registered on the instance before fitting, so the
        # estimator is rebuilt per series rather than shared.
        self._estimator = Prophet(**self.params)

    # Fit Prophet on the series, registering any usable feature regressors
    def train(self, series: ForecastSeries) -> dict[str, Any]:
        self.initialize()

        frame = series.frame
        training_frame = pd.DataFrame(
            {
                "ds": pd.to_datetime(frame[series.date_column]),
                "y": frame[series.target_column].astype(float).to_numpy(),
            }
        )

        regressors = self._usable_feature_columns(series)
        for column in regressors:
            training_frame[column] = frame[column].to_numpy()
            self._estimator.add_regressor(column)

        self._estimator.fit(training_frame)
        self._model = self._estimator

        # Retained so prediction can extend the timeline at the same grain
        # and reproduce any regressors the fit depends on.
        self._training_frame = training_frame
        self._regressors = regressors

        return {
            "observations": int(len(training_frame)),
            "regressors": regressors,
            "seasonality_mode": self.params.get("seasonality_mode"),
        }

    # Forecast horizon periods with Prophet's native intervals
    def predict(self, horizon: int, future_frame: pd.DataFrame | None = None) -> ForecastOutput:
        # The future timeline is extended at the median spacing observed in
        # training, which keeps the adapter grain-agnostic.
        self._require_trained()

        history = self._training_frame
        last_date = history["ds"].iloc[-1]
        step = history["ds"].diff().median()

        future_dates = [last_date + step * (offset + 1) for offset in range(horizon)]
        future = pd.DataFrame({"ds": future_dates})

        # Prophet refuses to predict unless every registered regressor is
        # present; use supplied future values where available, otherwise
        # carry the last observed value forward.
        for column in self._regressors:
            if future_frame is not None and column in future_frame.columns:
                values = list(future_frame[column].to_numpy()[:horizon])
                if len(values) < horizon:
                    values += [history[column].iloc[-1]] * (horizon - len(values))
                future[column] = values
            else:
                future[column] = history[column].iloc[-1]

        forecast = self._model.predict(future)

        return ForecastOutput(
            values=[float(value) for value in forecast["yhat"]],
            lower=[float(value) for value in forecast["yhat_lower"]],
            upper=[float(value) for value in forecast["yhat_upper"]],
        )
