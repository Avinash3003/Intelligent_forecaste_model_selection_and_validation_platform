"""Combines the three signals into one composite score per model.

The importance signal is produced by the explainability stage beforehand and
looked up here, never computed.

Ranking never runs across groups: each group's candidates are normalized and
scored only against each other, so one business key's ranking can never be
influenced by another's data.
"""

from __future__ import annotations

import time

from forecast_engine.config.ranking_config import RankingConfig
from forecast_engine.s06_evaluation.evaluation_report import EvaluationReport, EvaluationResult
from forecast_engine.s06_evaluation.metrics import ForecastMetrics
from forecast_engine.s07_explainability.explainability_report import ExplainabilityReport
from forecast_engine.s08_ranking.ranking_report import BacktestScoreBreakdown, RankedModel, RankingReport, ShapScoreBreakdown
from forecast_engine.s08_ranking.stability_scorer import ForecastStabilityScorer

# Lower-is-better metrics; ranking scores them inverted (1.0 = best of group).
_METRIC_FIELDS = ("mape", "wmape", "rmse", "mae", "smape")


class ModelRanker:
    """Ranks every surviving model, independently per forecast group."""

    # Wire up config and the stability scorer dependency
    def __init__(
        self,
        config: RankingConfig | None = None,
        stability_scorer: ForecastStabilityScorer | None = None,
    ) -> None:
        self._config = config or RankingConfig.default()
        self._stability_scorer = stability_scorer or ForecastStabilityScorer(self._config.stability_weights)

    # Rank every group's surviving models
    def rank_all(
        self,
        evaluation_report: EvaluationReport,
        explainability_report: ExplainabilityReport,
    ) -> RankingReport:
        started = time.perf_counter()
        report = RankingReport()

        for group_id, survivors in evaluation_report.surviving_by_group().items():
            try:
                report.rankings[group_id] = self._rank_group(group_id, survivors, explainability_report)
            except Exception:  # noqa: BLE001 - one group's ranking failure must not affect others
                report.rankings[group_id] = []

        report.duration_seconds = time.perf_counter() - started
        return report

    # Score and rank one group's survivors by composite score
    def _rank_group(
        self,
        group_id: str,
        survivors: list[EvaluationResult],
        explainability_report: ExplainabilityReport,
    ) -> list[RankedModel]:
        backtest_scores = self._score_backtest(survivors)

        forecasts = {result.model_name: result.forecast for result in survivors if result.forecast}
        stability_scores = self._stability_scorer.score_group(forecasts)

        weights = self._config.composite_weights
        weight_total = weights.backtest_weight + weights.stability_weight + weights.shap_weight

        ranked: list[RankedModel] = []
        for result in survivors:
            backtest = backtest_scores.get(result.model_name, BacktestScoreBreakdown())
            stability = stability_scores.get(result.model_name)
            if stability is None:
                continue

            explainability = explainability_report.for_pair(group_id, result.model_name)
            shap = (
                explainability.ranking_summary()
                if explainability is not None
                else _neutral_shap_breakdown("No explainability result is available for this pair.")
            )

            composite = (
                backtest.score * weights.backtest_weight
                + stability.score * weights.stability_weight
                + shap.score * weights.shap_weight
            )
            composite = composite / weight_total if weight_total else 0.0

            ranked.append(
                RankedModel(
                    group_id=group_id,
                    model_name=result.model_name,
                    key_values=result.key_values,
                    backtest=backtest,
                    stability=stability,
                    shap=shap,
                    composite_score=composite,
                )
            )

        # Original backtesting rank: independent of the composite, purely by
        # backtest accuracy — so the report can show how much ranking moved
        # a model away from a pure "lowest error wins" order.
        for rank, model in enumerate(sorted(ranked, key=lambda m: m.backtest.score, reverse=True), start=1):
            model.original_backtest_rank = rank

        ranked.sort(key=lambda m: m.composite_score, reverse=True)
        for rank, model in enumerate(ranked, start=1):
            model.final_composite_rank = rank

        return ranked

    # Normalize each backtest metric across survivors, then weight-combine
    def _score_backtest(self, survivors: list[EvaluationResult]) -> dict[str, BacktestScoreBreakdown]:
        metrics_by_model: dict[str, ForecastMetrics] = {
            result.model_name: result.backtest.overall
            for result in survivors
            if result.backtest and result.backtest.overall
        }

        normalized_by_metric: dict[str, dict[str, float]] = {}
        for field_name in _METRIC_FIELDS:
            raw = {
                name: getattr(metrics, field_name)
                for name, metrics in metrics_by_model.items()
                if getattr(metrics, field_name) is not None
            }
            normalized_by_metric[field_name] = _inverse_normalize(raw)

        weights = self._config.metric_weights
        weight_map = {
            "mape": weights.mape_weight,
            "wmape": weights.wmape_weight,
            "rmse": weights.rmse_weight,
            "mae": weights.mae_weight,
            "smape": weights.smape_weight,
        }

        results: dict[str, BacktestScoreBreakdown] = {}
        for name in metrics_by_model:
            per_model_normalized = {
                field_name: normalized_by_metric[field_name][name]
                for field_name in _METRIC_FIELDS
                if name in normalized_by_metric[field_name]
            }

            weighted_sum = sum(value * weight_map[field_name] for field_name, value in per_model_normalized.items())
            weight_total = sum(weight_map[field_name] for field_name in per_model_normalized)
            score = weighted_sum / weight_total if weight_total else 0.0

            results[name] = BacktestScoreBreakdown(normalized=per_model_normalized, score=score)

        # A survivor whose backtest was entirely skipped (too short a
        # series) gets a neutral mid-range score rather than being excluded
        # from ranking outright.
        for result in survivors:
            if result.model_name not in results:
                results[result.model_name] = BacktestScoreBreakdown(normalized={}, score=0.5)

        return results


# Safe default when an explainability result is unexpectedly missing
def _neutral_shap_breakdown(reason: str) -> ShapScoreBreakdown:
    # Neutral rather than punitive, for the same reason _score_backtest gives
    # a skipped backtest a mid-range score instead of excluding the model.
    return ShapScoreBreakdown(method="unavailable", score=0.5, skipped_reason=reason)


# Min-max normalize, then invert so a lower raw value scores higher
def _inverse_normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}

    low = min(values.values())
    high = max(values.values())

    if high == low:
        return {name: 1.0 for name in values}

    return {name: 1.0 - ((value - low) / (high - low)) for name, value in values.items()}
