"""Picks the model that ships, per forecast group.

Walks each group's ranked candidates best-first, running drift validation on
each until one passes or the list runs out. When every candidate fails —
including a group with no survivors at all — the fallback model is trained
fresh and used, clearly flagged as such.

Every group gets exactly one outcome, which is what the fallback path exists
for.
"""

from __future__ import annotations

import time

import pandas as pd

from forecast_engine.config.drift_config import DriftValidationConfig
from forecast_engine.config.model_config import ModelConfig
from forecast_engine.s09_drift.drift_validator import DriftValidator
from forecast_engine.s06_evaluation.evaluation_report import EvaluationReport, ForwardForecast
from forecast_engine.s05_models.model_registry import ModelRegistry
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.s08_ranking.ranking_report import RankedModel, RankingReport
from forecast_engine.s10_selection.selection_report import (
    FinalSelectionStatus,
    ProductionModelResult,
    ProductionSelectionReport,
    RejectedCandidate,
)


class ProductionModelSelector:
    """Selects the final production model for every forecasting group."""

    def __init__(
        self,
        registry: ModelRegistry,
        model_config: ModelConfig | None = None,
        drift_config: DriftValidationConfig | None = None,
        drift_validator: DriftValidator | None = None,
        forecast_horizon: int = 12,
    ) -> None:
        # Store dependencies and configuration
        self._registry = registry
        self._model_config = model_config or ModelConfig.default()
        self._drift_config = drift_config or DriftValidationConfig.default()
        self._validator = drift_validator or DriftValidator(self._drift_config)
        self._horizon = forecast_horizon

    def select_all(
        self,
        ranking_report: RankingReport,
        evaluation_report: EvaluationReport,
        series_collection: list[ForecastSeries],
    ) -> ProductionSelectionReport:
        """One production model per group.

        Groups are independent, and each returns exactly one result even when
        that result is "no model available".
        """
        # Select the final production model for every forecasting group
        started = time.perf_counter()
        report = ProductionSelectionReport()

        forecasts_by_key: dict[tuple[str, str], ForwardForecast] = {
            (result.group_id, result.model_name): result.forecast
            for result in evaluation_report.results
            if result.forecast is not None
        }

        for series in series_collection:
            try:
                candidates = ranking_report.for_group(series.group_id)
                report.results.append(self._select_one(series, candidates, forecasts_by_key))
            except Exception as exc:  # noqa: BLE001 - one group's failure must not affect others
                report.results.append(
                    ProductionModelResult(
                        group_id=series.group_id,
                        status=FinalSelectionStatus.FAILED,
                        key_values=series.key_values,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        report.duration_seconds = time.perf_counter() - started
        return report

    def _select_one(
        self,
        series: ForecastSeries,
        candidates: list[RankedModel],
        forecasts_by_key: dict[tuple[str, str], ForwardForecast],
    ) -> ProductionModelResult:
        # Sequential Candidate 1 -> Drift Validation -> Pass? loop
        rejected: list[RejectedCandidate] = []
        # Recorded before the loop so the fallback path can report exactly
        # which candidates were in play, even when the list was empty.
        original_candidates = [candidate.model_name for candidate in candidates]

        for candidate in candidates:
            forecast = forecasts_by_key.get((series.group_id, candidate.model_name))
            if forecast is None:
                rejected.append(
                    RejectedCandidate(candidate.model_name, "No forward forecast is available to validate.")
                )
                continue

            try:
                drift_result = self._validator.validate(series, forecast)
            except Exception as exc:  # noqa: BLE001 - one candidate's drift failure must not block the next
                rejected.append(RejectedCandidate(candidate.model_name, f"Drift validation error: {exc}"))
                continue

            if drift_result.passed:
                return ProductionModelResult(
                    group_id=series.group_id,
                    model_name=candidate.model_name,
                    key_values=series.key_values,
                    composite_score=candidate.composite_score,
                    original_backtest_rank=candidate.original_backtest_rank,
                    final_rank=candidate.final_composite_rank,
                    drift_validation=drift_result,
                    status=FinalSelectionStatus.SELECTED,
                    fallback_used=False,
                    rejected_candidates=rejected,
                    original_candidates=original_candidates,
                    forecast=forecast,
                )

            rejected.append(
                RejectedCandidate(
                    model_name=candidate.model_name,
                    reason=drift_result.detail,
                    algorithm=drift_result.algorithm_selection.algorithm.value,
                    statistic=round(drift_result.drift_statistic, 6),
                    threshold_method=drift_result.threshold.method.value,
                    threshold_value=round(drift_result.threshold.value, 6),
                )
            )

        # Every ranked candidate failed drift validation, or there were none
        # to begin with — Section 6.9's fallback path.
        return self._use_fallback(series, rejected, original_candidates)

    def _use_fallback(
        self,
        series: ForecastSeries,
        rejected: list[RejectedCandidate],
        original_candidates: list[str],
    ) -> ProductionModelResult:
        # Train and apply the configured fallback model
        fallback_name = self._model_config.fallback_model
        spec = self._registry.config.find(fallback_name)

        # One-line reason the fallback path ran, for the dashboard and the
        # LLM narrative; per-candidate detail stays in `rejected`.
        trigger = (
            f"All {len(original_candidates)} evaluated model(s) failed validation "
            f"({', '.join(original_candidates)})."
            if original_candidates
            else "No ranked candidate survived earlier validation stages."
        )

        if spec is None:
            return ProductionModelResult(
                group_id=series.group_id,
                key_values=series.key_values,
                status=FinalSelectionStatus.NO_MODEL_AVAILABLE,
                error=f"Configured fallback model '{fallback_name}' is not registered.",
                rejected_candidates=rejected,
                original_candidates=original_candidates,
                fallback_model=fallback_name,
                fallback_trigger=trigger,
            )

        try:
            model = self._registry.create(spec)
            model.initialize()
            model.train(series)
            output = model.predict(self._horizon)
        except Exception as exc:  # noqa: BLE001 - an unusable fallback must not fail the whole run
            return ProductionModelResult(
                group_id=series.group_id,
                key_values=series.key_values,
                status=FinalSelectionStatus.NO_MODEL_AVAILABLE,
                error=f"Fallback model '{fallback_name}' could not be trained: {type(exc).__name__}: {exc}",
                rejected_candidates=rejected,
                original_candidates=original_candidates,
                fallback_model=fallback_name,
                fallback_trigger=trigger,
            )

        forecast = ForwardForecast(
            dates=[_iso(d) for d in _future_dates(series, self._horizon)],
            values=list(output.values),
            lower=output.lower,
            upper=output.upper,
        )

        return ProductionModelResult(
            group_id=series.group_id,
            model_name=fallback_name,
            key_values=series.key_values,
            status=FinalSelectionStatus.FALLBACK_USED,
            fallback_used=True,
            # Carried so persistence can serialize the estimator this path
            # already fitted, instead of fitting an identical one again.
            fitted_model=model,
            rejected_candidates=rejected,
            original_candidates=original_candidates,
            fallback_model=fallback_name,
            fallback_trigger=trigger,
            forecast=forecast,
        )


# Extend the timeline beyond the last observation, at the median observed spacing
def _future_dates(series: ForecastSeries, horizon: int) -> list[pd.Timestamp]:
    # Small, self-contained mirror of ForwardForecastGenerator._future_dates —
    # duplicated deliberately rather than imported, since reaching into another
    # package's protected method would couple this stage to internals it has
    # no business depending on.
    dates = pd.to_datetime(series.frame[series.date_column])
    last_date = dates.iloc[-1]

    spacing = dates.diff().median()
    if pd.isna(spacing) or spacing == pd.Timedelta(0):
        spacing = pd.Timedelta(days=30)

    return [last_date + spacing * (offset + 1) for offset in range(horizon)]


# Format a timestamp as ISO 8601
def _iso(value: pd.Timestamp) -> str:
    return value.isoformat()
