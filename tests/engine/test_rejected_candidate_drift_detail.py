"""A rejected candidate's own drift statistic/threshold must survive into
the final result, not just the formatted `reason` sentence.

Before this fix, `production_selector.py` discarded the full
`DriftValidationResult` after formatting it into `RejectedCandidate.reason`
— the numbers existed (drift_validator.py's `detail` text literally states
them) but were never stored as real fields. The Results dashboard's "Drift
statistic"/"Drift threshold" columns had nothing to read for a rejected
model and always showed "—", even though the model's own rejection reason
was "Failed drift validation".
"""

from __future__ import annotations

import pandas as pd

from forecast_engine.config.drift_config import DriftAlgorithm, ThresholdMethod
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.s06_evaluation.evaluation_report import ForwardForecast
from forecast_engine.s08_ranking.ranking_report import RankedModel, RankingReport
from forecast_engine.s09_drift.drift_report import DriftAlgorithmSelection, DriftValidationResult, ThresholdEstimate
from forecast_engine.s10_selection.production_selector import ProductionModelSelector
from forecast_engine.s10_selection.selection_report import FinalSelectionStatus


def _series(group_id: str) -> ForecastSeries:
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=6, freq="MS"), "target": [1, 2, 3, 4, 5, 6]})
    return ForecastSeries(group_id=group_id, frame=frame, date_column="date", target_column="target")


def _ranked(group_id: str, model_name: str, rank: int) -> RankedModel:
    return RankedModel(group_id=group_id, model_name=model_name, original_backtest_rank=rank, final_composite_rank=rank)


def _forecast() -> ForwardForecast:
    return ForwardForecast(dates=["2020-07-01"], values=[7.0])


def _drift_result(passed: bool, statistic: float, threshold: float) -> DriftValidationResult:
    return DriftValidationResult(
        algorithm_selection=DriftAlgorithmSelection(
            algorithm=DriftAlgorithm.POPULATION_STABILITY_INDEX, reason="test", sample_size=10, unique_values=5
        ),
        threshold=ThresholdEstimate(method=ThresholdMethod.PERCENTILE, value=threshold, reason="test"),
        drift_statistic=statistic,
        passed=passed,
        detail=f"population_stability_index statistic {statistic:.6f} > threshold {threshold:.6f} (percentile).",
    )


class _StubRegistry:
    """Just enough for the fallback path to resolve "no model registered"
    cleanly (production_selector.py's `spec is None` branch) without
    actually training anything - these tests only care about what lands in
    `rejected_candidates`, which is built before the fallback path runs."""

    class _Config:
        def find(self, name):
            return None

    config = _Config()


class _StubValidator:
    """Returns a queued, per-candidate DriftValidationResult in call order —
    a real per-model drift statistic each time, not one shared value, so a
    test that only checked "some number appears" couldn't hide a bug where
    every rejected candidate got the same (wrong) number."""

    def __init__(self, results: list[DriftValidationResult]) -> None:
        self._results = list(results)

    def validate(self, series, forecast):
        return self._results.pop(0)


def test_each_rejected_candidates_own_drift_numbers_are_preserved():
    ranking = RankingReport(rankings={"g1": [_ranked("g1", "xgboost", 1), _ranked("g1", "arima", 2)]})
    # Both fail (passed=False) with deliberately distinct numbers, so the
    # test can catch a bug where every rejected candidate ends up sharing
    # one (wrong) statistic/threshold instead of keeping its own.
    validator = _StubValidator(
        [
            _drift_result(passed=False, statistic=0.183245, threshold=0.15),
            _drift_result(passed=False, statistic=0.09, threshold=0.12),
        ]
    )

    selector = ProductionModelSelector(registry=_StubRegistry(), drift_validator=validator, forecast_horizon=1)
    result = selector._select_one(_series("g1"), ranking.for_group("g1"), {
        ("g1", "xgboost"): _forecast(),
        ("g1", "arima"): _forecast(),
    })

    assert result.status is FinalSelectionStatus.NO_MODEL_AVAILABLE
    rejected = {c.model_name: c for c in result.rejected_candidates}

    assert rejected["xgboost"].statistic == 0.183245
    assert rejected["xgboost"].threshold_value == 0.15
    assert rejected["xgboost"].algorithm == "population_stability_index"
    assert rejected["xgboost"].threshold_method == "percentile"

    assert rejected["arima"].statistic == 0.09
    assert rejected["arima"].threshold_value == 0.12
    # Each candidate keeps its own numbers - arima's are not overwritten by
    # or copied from xgboost's, which a shared-mutable-object bug could do.
    assert rejected["xgboost"].statistic != rejected["arima"].statistic


def test_a_candidate_with_no_forward_forecast_has_no_drift_numbers():
    """This candidate never reached drift validation at all - None here
    means "not evaluated", never a false zero."""
    ranking = RankingReport(rankings={"g1": [_ranked("g1", "xgboost", 1)]})
    selector = ProductionModelSelector(registry=_StubRegistry(), drift_validator=_StubValidator([]), forecast_horizon=1)

    result = selector._select_one(_series("g1"), ranking.for_group("g1"), {})

    rejected = result.rejected_candidates
    assert len(rejected) == 1
    assert rejected[0].model_name == "xgboost"
    assert rejected[0].statistic is None
    assert rejected[0].threshold_value is None
    assert "No forward forecast" in rejected[0].reason


def test_rejected_candidate_to_dict_uses_the_same_keys_as_the_winners_drift_fields():
    """selection_report.py's ProductionModelResult.to_dict() names the
    winner's drift fields selected_drift_algorithm/drift_statistic/
    dynamic_threshold_method/dynamic_threshold_value - a rejected
    candidate's to_dict() must use the exact same keys so a downstream
    consumer (result_service.py's _drift_detail) reads both the same way."""
    from forecast_engine.s10_selection.selection_report import RejectedCandidate

    candidate = RejectedCandidate(
        model_name="xgboost",
        reason="population_stability_index statistic 0.183245 > threshold 0.150000 (percentile).",
        algorithm="population_stability_index",
        statistic=0.183245,
        threshold_method="percentile",
        threshold_value=0.15,
    )
    payload = candidate.to_dict()

    assert payload["selected_drift_algorithm"] == "population_stability_index"
    assert payload["drift_statistic"] == 0.183245
    assert payload["dynamic_threshold_method"] == "percentile"
    assert payload["dynamic_threshold_value"] == 0.15
    assert payload["reason"] == candidate.reason
