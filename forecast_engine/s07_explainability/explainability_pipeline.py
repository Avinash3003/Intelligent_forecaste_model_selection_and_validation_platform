"""Explainability orchestration — Phase 8 end to end.

Generates SHAP/importance results for every surviving (group, model) pair
from Phase 7A, mirroring `EvaluationPipeline`'s per-pair failure isolation.
Runs strictly before Model Ranking, whose composite score consumes this
report's output rather than computing importance itself.
"""

from __future__ import annotations

import time

from forecast_engine.config.explainability_config import ExplainabilityConfig
from forecast_engine.s06_evaluation.evaluation_report import EvaluationReport
from forecast_engine.s07_explainability.explainability_report import ExplainabilityReport
from forecast_engine.s07_explainability.shap_engine import ShapExplainabilityEngine
from forecast_engine.s05_models.base_model import TrainedModel, TrainingStatus
from forecast_engine.s05_models.model_registry import ModelRegistry
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries


class ExplainabilityPipeline:
    """Generates explainability results for every surviving model."""

    # Wire up (or default-construct) the explainability engine
    def __init__(
        self,
        registry: ModelRegistry,
        config: ExplainabilityConfig | None = None,
        engine: ShapExplainabilityEngine | None = None,
    ) -> None:
        self._config = config or ExplainabilityConfig.default()
        self._engine = engine or ShapExplainabilityEngine(registry, self._config)

    # Generate one ExplainabilityResult per surviving (group, model) pair
    def generate_all(
        self,
        evaluation_report: EvaluationReport,
        trained_models: list[TrainedModel],
        series_collection: list[ForecastSeries],
    ) -> ExplainabilityReport:
        started = time.perf_counter()
        report = ExplainabilityReport()

        trained_by_key = {
            (m.group_id, m.model_name): m for m in trained_models if m.status is TrainingStatus.TRAINED
        }
        series_by_group = {series.group_id: series for series in series_collection}

        for result in evaluation_report.surviving_models():
            trained = trained_by_key.get((result.group_id, result.model_name))
            series = series_by_group.get(result.group_id)
            if trained is None or series is None:
                continue
            report.results.append(self._engine.compute(trained, series))

        report.duration_seconds = time.perf_counter() - started
        return report
