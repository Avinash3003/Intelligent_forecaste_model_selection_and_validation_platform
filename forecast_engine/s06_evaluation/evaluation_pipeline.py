"""Sequences evaluation for every trained model.

    backtest -> metrics -> forward forecast -> forward validation

and returns the survivors. It decides nothing about which model is best —
that is ranking's job; this only produces the eligible candidates.

Every (group, model) pair is guarded independently, so a model that cannot
be backtested or produces an implausible forecast is recorded and skipped
while the rest proceed.
"""

from __future__ import annotations

import time

from forecast_engine.config.evaluation_config import EvaluationConfig
from forecast_engine.s06_evaluation.backtest_engine import BacktestEngine
from forecast_engine.s06_evaluation.evaluation_report import (
    EvaluationReport,
    EvaluationResult,
    EvaluationStatus,
)
from forecast_engine.s06_evaluation.forecast_generator import ForwardForecastGenerator
from forecast_engine.s06_evaluation.forward_validator import ForwardForecastValidator
from forecast_engine.s05_models.base_model import TrainedModel, TrainingStatus
from forecast_engine.s05_models.model_registry import ModelRegistry
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.utils.exceptions import ForecastEngineError


class EvaluationPipeline:
    """Backtests, forecasts and validates every trained model."""

    # Wire up (or default-construct) the four stage collaborators
    def __init__(
        self,
        registry: ModelRegistry,
        config: EvaluationConfig | None = None,
        backtest_engine: BacktestEngine | None = None,
        forecast_generator: ForwardForecastGenerator | None = None,
        validator: ForwardForecastValidator | None = None,
    ) -> None:
        self._config = config or EvaluationConfig.default()
        self._registry = registry
        self._backtest_engine = backtest_engine or BacktestEngine(
            registry, self._config.backtest, self._config.metrics
        )
        self._forecast_generator = forecast_generator or ForwardForecastGenerator(
            registry, self._config.forecast_horizon
        )
        self._validator = validator or ForwardForecastValidator(self._config.forward_validation)

    # Evaluate every trained model against its own forecasting group
    def evaluate_all(
        self, series_collection: list[ForecastSeries], trained_models: list[TrainedModel]
    ) -> EvaluationReport:
        started = time.perf_counter()
        report = EvaluationReport()

        series_by_group = {series.group_id: series for series in series_collection}

        # A model that never trained has nothing to evaluate; skipping it
        # here keeps the report aligned with what actually exists.
        evaluable = [model for model in trained_models if model.status is TrainingStatus.TRAINED]

        for trained in evaluable:
            series = series_by_group.get(trained.group_id)
            if series is None:
                report.results.append(
                    EvaluationResult(
                        group_id=trained.group_id,
                        model_name=trained.model_name,
                        status=EvaluationStatus.SKIPPED,
                        key_values=trained.key_values,
                        error="No series is available for this forecasting group.",
                    )
                )
                continue

            result, timing = self._evaluate_one(trained, series)
            report.results.append(result)
            report.backtest_seconds += timing["backtest_seconds"]
            report.forecast_generation_seconds += timing["forecast_seconds"]
            report.validation_seconds += timing["validation_seconds"]
            report.model_fit_count += timing["fit_count"]
            report.backtest_windows_evaluated += timing["backtest_windows"]
            if timing["forecast_reused"]:
                report.forecasts_reused += 1
            elif timing["forecast_attempted"]:
                report.forecasts_refit += 1

        report.groups_evaluated = len({result.group_id for result in report.results})
        report.duration_seconds = time.perf_counter() - started
        return report

    # Run all four stages for one group/model pair; never raises. Returns
    # the result plus a dict of this pair's own timing/fit-count
    # contribution, so the caller can accumulate stage-level telemetry
    # without this method reaching into the shared report directly.
    def _evaluate_one(
        self, trained: TrainedModel, series: ForecastSeries
    ) -> tuple[EvaluationResult, dict[str, float | int | bool]]:
        result = EvaluationResult(
            group_id=trained.group_id,
            model_name=trained.model_name,
            status=EvaluationStatus.FAILED,
            key_values=trained.key_values,
        )
        timing = {
            "backtest_seconds": 0.0,
            "forecast_seconds": 0.0,
            "validation_seconds": 0.0,
            "fit_count": 0,
            "backtest_windows": 0,
            "forecast_attempted": False,
            "forecast_reused": False,
        }

        try:
            # 1–2. Backtest and metrics. A series too short to split yields
            # a skipped backtest, not a failure — the model can still be
            # forecast and validated, and ranking decides what thin
            # historical evidence is worth.
            #
            # Every fold here is a genuinely fresh fit (Section 6.4,
            # "honest refitting") — a fold's score would leak future data
            # into itself if it reused any fit made on a wider window, so
            # none of this work is a candidate for reuse.
            backtest_started = time.perf_counter()
            result.backtest = self._backtest_engine.run(trained, series)
            timing["backtest_seconds"] = time.perf_counter() - backtest_started
            timing["backtest_windows"] = result.backtest.window_count
            timing["fit_count"] += result.backtest.window_count

            # 3. Forecast the full horizon. Reuses the model Training
            # already fit on this exact series (same series, same tuned
            # params) instead of fitting an identical model a second time —
            # see ForwardForecastGenerator.generate() for why that refit
            # was a verbatim duplicate, not independent evidence.
            forecast_started = time.perf_counter()
            timing["forecast_attempted"] = True
            result.forecast = self._forecast_generator.generate(trained, series)
            timing["forecast_seconds"] = time.perf_counter() - forecast_started
            timing["forecast_reused"] = self._forecast_generator.last_call_reused_fit
            if not timing["forecast_reused"]:
                timing["fit_count"] += 1

            # 4. Elimination framework.
            validation_started = time.perf_counter()
            result.validation = self._validator.validate(series, result.forecast)
            timing["validation_seconds"] = time.perf_counter() - validation_started
            result.status = (
                EvaluationStatus.SURVIVED if result.validation.passed else EvaluationStatus.ELIMINATED
            )
        except ForecastEngineError as exc:
            result.status = EvaluationStatus.FAILED
            result.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - isolated per pair by design
            result.status = EvaluationStatus.FAILED
            result.error = f"{type(exc).__name__}: {exc}"

        return result, timing
