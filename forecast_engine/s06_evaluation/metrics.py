"""The five accuracy metrics: MAPE, WMAPE, RMSE, MAE, SMAPE.

Pure functions that know nothing about models, groups or backtesting, so the
same code scores a fold and anything else.

The percentage metrics need care:
  - MAPE divides by the actual and is undefined at zero; those observations
    are excluded rather than producing an infinity that poisons aggregates.
  - WMAPE is a ratio of sums, so one zero cannot destabilise it — which is
    why platform accuracy is defined as 100% - WMAPE.
  - SMAPE is undefined only when both sides are zero, treated as perfect
    agreement rather than dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from forecast_engine.config.evaluation_config import MetricsConfig


@dataclass
class ForecastMetrics:
    """The five metrics for one actual/predicted comparison.

    None means "not measurable here" (MAPE on all-zero actuals, say), which
    a caller must handle differently from a genuine zero error.
    """

    mape: float | None = None
    wmape: float | None = None
    rmse: float | None = None
    mae: float | None = None
    smape: float | None = None
    observations: int = 0

    # Platform accuracy, defined as 100% − WMAPE (Section 10)
    @property
    def accuracy(self) -> float | None:
        if self.wmape is None:
            return None
        return max(0.0, 100.0 - self.wmape)

    # Serialize metrics to a rounded dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "mape": _round(self.mape),
            "wmape": _round(self.wmape),
            "rmse": _round(self.rmse),
            "mae": _round(self.mae),
            "smape": _round(self.smape),
            "accuracy": _round(self.accuracy),
            "observations": self.observations,
        }


# Round a metric value, passing through None
def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


class MetricsCalculator:
    """Computes ForecastMetrics from aligned actual/predicted values."""

    def __init__(self, config: MetricsConfig | None = None) -> None:
        self._config = config or MetricsConfig()

    # Score predicted values against actuals, returning all five metrics
    def calculate(self, actual: list[float] | np.ndarray, predicted: list[float] | np.ndarray) -> ForecastMetrics:
        actual_array = np.asarray(actual, dtype=float)
        predicted_array = np.asarray(predicted, dtype=float)

        if actual_array.size == 0 or actual_array.size != predicted_array.size:
            return ForecastMetrics()

        # A non-finite prediction cannot be scored; excluding the pair keeps
        # one bad step from nullifying an otherwise measurable window.
        usable = np.isfinite(actual_array) & np.isfinite(predicted_array)
        actual_array = actual_array[usable]
        predicted_array = predicted_array[usable]

        if actual_array.size == 0:
            return ForecastMetrics()

        errors = actual_array - predicted_array
        absolute_errors = np.abs(errors)

        return ForecastMetrics(
            mape=self._mape(actual_array, absolute_errors),
            wmape=self._wmape(actual_array, absolute_errors),
            rmse=float(math.sqrt(float(np.mean(errors**2)))),
            mae=float(np.mean(absolute_errors)),
            smape=self._smape(actual_array, predicted_array, absolute_errors),
            observations=int(actual_array.size),
        )

    # Mean absolute percentage error, excluding zero actuals
    def _mape(self, actual: np.ndarray, absolute_errors: np.ndarray) -> float | None:
        if self._config.exclude_zero_actuals_from_mape:
            nonzero = actual != 0
            if not nonzero.any():
                return None
            return float(np.mean(absolute_errors[nonzero] / np.abs(actual[nonzero])) * 100)

        if (actual == 0).any():
            return None
        return float(np.mean(absolute_errors / np.abs(actual)) * 100)

    # Weighted MAPE — total absolute error over total actual volume
    def _wmape(self, actual: np.ndarray, absolute_errors: np.ndarray) -> float | None:
        denominator = float(np.sum(np.abs(actual)))
        if denominator == 0:
            return None
        return float(np.sum(absolute_errors) / denominator * 100)

    # Symmetric MAPE, treating 0-vs-0 as exact agreement
    def _smape(
        self, actual: np.ndarray, predicted: np.ndarray, absolute_errors: np.ndarray
    ) -> float | None:
        denominator = (np.abs(actual) + np.abs(predicted)) / 2

        # Where both sides are zero the forecast is exactly right; scoring
        # it as zero error is more faithful than discarding the point.
        ratio = np.zeros_like(denominator, dtype=float)
        measurable = denominator != 0
        if not measurable.any():
            return 0.0

        ratio[measurable] = absolute_errors[measurable] / denominator[measurable]
        return float(np.mean(ratio) * 100)
