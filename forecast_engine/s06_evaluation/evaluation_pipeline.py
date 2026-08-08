"""Evaluation orchestration — Phase 7A end to end.

Sequences the four stages this phase owns for every trained model:

    Rolling Backtesting -> Metrics -> Forward Forecast -> Forward Validation

and returns the surviving models. It decides nothing about which model is
*best*: Section 6.5's output is the set of candidates eligible for ranking,
and ranking itself is the next phase's work.

Failure isolation is the defining behaviour. Every (group, model) pair is
guarded independently, so a model that cannot be backtested, cannot
forecast, or produces an implausible forecast is recorded and skipped while
every other pair proceeds — Section 5.3 requires failed keys to be flagged
without blocking the rest of the run.
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

            report.results.append(self._evaluate_one(trained, series))

        report.groups_evaluated = len({result.group_id for result in report.results})
        report.duration_seconds = time.perf_counter() - started
        return report

    # Run all four stages for one group/model pair; never raises
    def _evaluate_one(self, trained: TrainedModel, series: ForecastSeries) -> EvaluationResult:
        result = EvaluationResult(
            group_id=trained.group_id,
            model_name=trained.model_name,
            status=EvaluationStatus.FAILED,
            key_values=trained.key_values,
        )

        try:
            # 1–2. Backtest and metrics. A series too short to split yields
            # a skipped backtest, not a failure — the model can still be
            # forecast and validated, and ranking decides what thin
            # historical evidence is worth.
            result.backtest = self._backtest_engine.run(trained, series)

            # 3. Refit on the complete series and forecast the full horizon.
            result.forecast = self._forecast_generator.generate(trained, series)

            # 4. Elimination framework.
            result.validation = self._validator.validate(series, result.forecast)
            result.status = (
                EvaluationStatus.SURVIVED if result.validation.passed else EvaluationStatus.ELIMINATED
            )
        except ForecastEngineError as exc:
            result.status = EvaluationStatus.FAILED
            result.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - isolated per pair by design
            result.status = EvaluationStatus.FAILED
            result.error = f"{type(exc).__name__}: {exc}"

        return result
