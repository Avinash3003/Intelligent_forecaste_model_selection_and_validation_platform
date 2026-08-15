"""Sequences ranking and final selection.

    rank -> select algorithm -> estimate threshold -> validate -> pick winner

The importance input is produced by the explainability stage and handed in
finished; this does not generate it.

Ranker and selector each isolate failures per group already, so this only
sequences the two into one call.
"""

from __future__ import annotations

from forecast_engine.config.drift_config import DriftValidationConfig
from forecast_engine.config.model_config import ModelConfig
from forecast_engine.config.ranking_config import RankingConfig
from forecast_engine.s09_drift.drift_validator import DriftValidator
from forecast_engine.s06_evaluation.evaluation_report import EvaluationReport
from forecast_engine.s07_explainability.explainability_report import ExplainabilityReport
from forecast_engine.s05_models.model_registry import ModelRegistry
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.s08_ranking.model_ranker import ModelRanker
from forecast_engine.s08_ranking.ranking_report import RankingReport
from forecast_engine.s10_selection.production_selector import ProductionModelSelector
from forecast_engine.s10_selection.selection_report import ProductionSelectionReport


class ProductionSelectionPipeline:
    """Ranks Phase 7A's survivors and selects the final production model."""

    def __init__(
        self,
        registry: ModelRegistry,
        ranking_config: RankingConfig | None = None,
        drift_config: DriftValidationConfig | None = None,
        model_config: ModelConfig | None = None,
        ranker: ModelRanker | None = None,
        selector: ProductionModelSelector | None = None,
        forecast_horizon: int = 12,
    ) -> None:
        # Store dependencies and construct default ranker/selector
        self._registry = registry
        self._ranking_config = ranking_config or RankingConfig.default()
        self._drift_config = drift_config or DriftValidationConfig.default()
        self._model_config = model_config or ModelConfig.default()

        self._ranker = ranker or ModelRanker(self._ranking_config)
        self._selector = selector or ProductionModelSelector(
            registry,
            self._model_config,
            self._drift_config,
            DriftValidator(self._drift_config),
            forecast_horizon,
        )

    def run(
        self,
        evaluation_report: EvaluationReport,
        explainability_report: ExplainabilityReport,
        series_collection: list[ForecastSeries],
    ) -> tuple[RankingReport, ProductionSelectionReport]:
        """Rank the survivors, then pick a winner per group.

        One stage because selection walks the ranked order best-first until a
        candidate passes drift validation; splitting them would mean
        persisting an intermediate ranking nothing else reads.
        """
        # Rank every group's survivors, then select each group's final production model
        ranking_report = self._ranker.rank_all(evaluation_report, explainability_report)
        selection_report = self._selector.select_all(ranking_report, evaluation_report, series_collection)
        return ranking_report, selection_report
