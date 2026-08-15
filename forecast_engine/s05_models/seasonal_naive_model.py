"""The fallback model, used when every ranked candidate fails drift validation.

Seasonal naive is the standard choice: no hyperparameters to mistune, cannot
fail to converge, and its forecast is simply "what happened at this point in
the cycle last time" — a transparent baseline, not another competitor.

Registered with enabled=False, so it is never trained as a normal candidate;
final selection looks it up by name only when it is needed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from forecast_engine.s05_models.base_model import BaseForecastingModel, ForecastOutput
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries


class SeasonalNaiveModel(BaseForecastingModel):
    """Repeats each period's value from one seasonal cycle earlier.

    Falls back to repeating the last observed value when the series is
    shorter than a full cycle, so a forecast is always produced.
    """

    # No external library, so always available
    @classmethod
    def is_available(cls) -> bool:
        return True

    # Validate the configured seasonal period before any data is touched
    def initialize(self) -> None:
        period = int(self.params.get("seasonal_period", 12))
        if period < 1:
            raise ValueError(f"seasonal_period must be at least 1; got {period}.")
        self.params["seasonal_period"] = period

    # Remember the last seasonal cycle (or last value) of the target series
    def train(self, series: ForecastSeries) -> dict[str, Any]:
        target = self._target_series(series)
        period = self.params["seasonal_period"]
        history = target.to_numpy(dtype=float)

        # The model *is* the tail of history: no fitting beyond remembering
        # the last full cycle (or the last value, if history is shorter).
        # Stored as `self._model` — same slot every other adapter uses —
        # so `TrainedModel.model`/`is_trained` behave consistently.
        self._model = history
        self._effective_period = period if len(history) >= period else 1

        return {
            "observations": int(len(history)),
            "seasonal_period": period,
            "effective_period": self._effective_period,
        }

    # Cycle through the last effective_period observed values
    def predict(self, horizon: int, future_frame: pd.DataFrame | None = None) -> ForecastOutput:
        # No native uncertainty quantification exists for this method, so
        # intervals are left unset rather than fabricated.
        self._require_trained()

        cycle = self._model[-self._effective_period :]
        values = [float(cycle[step % len(cycle)]) for step in range(horizon)]
        return ForecastOutput(values=values)
