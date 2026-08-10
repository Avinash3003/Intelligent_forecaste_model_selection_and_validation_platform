"""Twelve-month forward forecast generation (Section 6.4 → 6.5).

The forward forecast must come from a model that has seen the group's
*complete* history — backtesting only ever fits on partial windows, and a
forecast that will be validated and potentially promoted needs every
available observation, not a fold's subset.

That model already exists by the time this runs: Model Training (Section
6.3.3) already fit this exact (group, model) pair on this exact series with
these exact tuned parameters. Fitting a second, identical model here would
be a verbatim duplicate — same inputs, same deterministic outcome, pure
wasted compute — so this reuses the trained wrapper's own `.predict()`
whenever the caller supplies one (`TrainedModel.fitted_model`). It falls
back to fitting fresh only when no fitted wrapper is available, which
keeps this class correct on its own rather than silently depending on the
caller always providing one.

The future timeline is extended at the median spacing observed in the
series, so the generator stays grain-agnostic — it produces monthly dates
for the monthly data the platform forecasts on without hard-coding that
assumption.

Confidence intervals are carried through only where the model produces them
natively (Section 5.6 draws a band when one exists); nothing is fabricated
for the models that do not.
"""

from __future__ import annotations

import pandas as pd

from forecast_engine.s06_evaluation.evaluation_report import ForwardForecast
from forecast_engine.s05_models.base_model import TrainedModel
from forecast_engine.s05_models.model_registry import ModelRegistry
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.utils.exceptions import ForecastGenerationError


class ForwardForecastGenerator:
    """Refits a model on full history and produces its forward forecast."""

    # Wire up the model registry and forecast horizon
    def __init__(self, registry: ModelRegistry, horizon: int = 12) -> None:
        self._registry = registry
        self._horizon = horizon
        # Set by the most recent `generate()` call, read by
        # `EvaluationPipeline` for its fit-count telemetry. Not thread-safe
        # by itself — fine today since evaluation runs one pair at a time
        # through a shared generator; a future parallel evaluator would need
        # `generate()` to return this instead of stashing it here.
        self.last_call_reused_fit: bool = False

    # Produce the forward forecast for one group/model pair
    def generate(self, trained: TrainedModel, series: ForecastSeries) -> ForwardForecast:
        self.last_call_reused_fit = False

        try:
            if trained.fitted_model is not None:
                self.last_call_reused_fit = True
                output = trained.fitted_model.predict(self._horizon)
            else:
                # No fitted wrapper was supplied (a caller outside the
                # normal pipeline, or a future code path) — fit fresh so
                # this class is correct standing alone, not only when
                # ModelTrainer happens to have populated `fitted_model`.
                spec = self._registry.config.find(trained.model_name)
                if spec is None:
                    raise ForecastGenerationError(
                        f"Model '{trained.model_name}' is no longer registered and cannot forecast."
                    )
                model = self._registry.create(spec, trained.params)
                model.initialize()
                model.train(series)
                output = model.predict(self._horizon)
        except ForecastGenerationError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
            raise ForecastGenerationError(
                f"'{trained.model_name}' failed to produce a forward forecast: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # Section 3 requires every surviving model to cover the full
        # horizon; a short forecast is a defect, not something to pad over.
        if len(output.values) < self._horizon:
            raise ForecastGenerationError(
                f"'{trained.model_name}' produced {len(output.values)} period(s) "
                f"but the configured horizon is {self._horizon}."
            )

        values = [float(value) for value in output.values[: self._horizon]]

        return ForwardForecast(
            dates=self._future_dates(series),
            values=values,
            lower=self._trim(output.lower),
            upper=self._trim(output.upper),
        )

    # Extend the timeline beyond the last observation using median spacing
    def _future_dates(self, series: ForecastSeries) -> list[str]:
        dates = pd.to_datetime(series.frame[series.date_column])
        last_date = dates.iloc[-1]

        spacing = dates.diff().median()
        if pd.isna(spacing) or spacing == pd.Timedelta(0):
            # A single observation gives nothing to measure; fall back to a
            # month, the grain the platform forecasts at.
            spacing = pd.Timedelta(days=30)

        return [(last_date + spacing * (offset + 1)).isoformat() for offset in range(self._horizon)]

    # Clip an interval bound to the horizon, or pass through None
    def _trim(self, bound: list[float] | None) -> list[float] | None:
        if bound is None:
            return None
        if len(bound) < self._horizon:
            return None
        return [float(value) for value in bound[: self._horizon]]
