"""ARIMA / SARIMA adapter (statsmodels).

Statistical models are selected differently from tree models. Rather than
scoring candidates on held-out data, ARIMA orders are chosen by an
information criterion (AIC) computed from the fit itself — the standard
approach for this family, and one that needs no validation split.

The candidate orders come from the registry's `search_space`; when the
space is empty the configured default order is used as-is, which is the
"configuration-driven parameter selection" the design calls for.
"""

from __future__ import annotations

import importlib.util
import itertools
import warnings
from typing import Any

import pandas as pd

from forecast_engine.s05_models.base_model import BaseForecastingModel, ForecastOutput
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries


class ArimaModel(BaseForecastingModel):
    """Univariate ARIMA, order selected by AIC."""

    # Whether statsmodels is importable
    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("statsmodels") is not None

    # Validate the configured order before any data is touched
    def initialize(self) -> None:
        order = tuple(self.params.get("order", (1, 1, 1)))
        if len(order) != 3:
            raise ValueError(f"ARIMA order must have three terms (p, d, q); got {order!r}.")
        self.params["order"] = order

    # ARIMA selects its own order internally via AIC, so tuning is skipped
    def supports_tuning(self) -> bool:
        return False

    # Fit candidate ARIMA orders and keep the one with the lowest AIC
    def train(self, series: ForecastSeries) -> dict[str, Any]:
        from statsmodels.tsa.arima.model import ARIMA

        target = self._target_series(series)
        candidate_orders = self._candidate_orders()

        best_fit = None
        best_order = None
        best_aic = float("inf")
        evaluated = 0

        # statsmodels emits convergence and date-index warnings freely on
        # short or irregular series; they are not actionable per-candidate,
        # and a failed candidate is simply skipped below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for order in candidate_orders:
                try:
                    fitted = ARIMA(target, order=order, trend=self.params.get("trend")).fit()
                except Exception:  # noqa: BLE001 - non-converging orders are expected
                    continue

                evaluated += 1
                if fitted.aic < best_aic:
                    best_aic, best_fit, best_order = fitted.aic, fitted, order

        if best_fit is None:
            raise ValueError(
                f"No ARIMA order could be fitted to this series "
                f"({len(candidate_orders)} candidate order(s) attempted)."
            )

        self._model = best_fit
        self.params["order"] = best_order

        return {
            "observations": int(len(target)),
            "selected_order": list(best_order),
            "aic": float(best_aic),
            "orders_evaluated": evaluated,
            "selection_criterion": "AIC",
        }

    # Forecast horizon steps ahead with native confidence intervals
    def predict(self, horizon: int, future_frame: pd.DataFrame | None = None) -> ForecastOutput:
        # ARIMA is univariate here, so future_frame carries no exogenous
        # values it could use; accepted only to satisfy the shared interface.
        self._require_trained()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._model.get_forecast(steps=horizon)
            predicted = forecast.predicted_mean
            intervals = forecast.conf_int()

        # conf_int() returns a two-column frame ordered [lower, upper]; its
        # column names embed the target name, so position is the stable way
        # to read it.
        lower = intervals.iloc[:, 0].to_numpy()
        upper = intervals.iloc[:, 1].to_numpy()

        return ForecastOutput(
            values=[float(value) for value in predicted],
            lower=[float(value) for value in lower],
            upper=[float(value) for value in upper],
        )

    # Build the (p, d, q) orders to evaluate
    def _candidate_orders(self) -> list[tuple[int, int, int]]:
        # Falls back to the single configured default when no search space is
        # declared, so disabling the search still yields a valid model.
        space = self.spec.search_space
        if not space:
            return [tuple(self.params["order"])]

        return [
            (p, d, q)
            for p, d, q in itertools.product(
                space.get("p", [1]), space.get("d", [1]), space.get("q", [1])
            )
        ]
