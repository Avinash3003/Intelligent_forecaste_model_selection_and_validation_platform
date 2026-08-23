"""A rejected model's own drift statistic/threshold must reach the Results
page's Model Decision table, not just its "Rejected" reason text.

Before this fix, `_drift_detail` only populated DriftDetail.statistic/
threshold_value for the winning model; a rejected candidate's DriftDetail
carried `detail` (free text) but no structured numbers, even when
`rejected_reasons[name]` (built from RejectedCandidate.to_dict()) already
had them — the Results table's "Drift statistic"/"Drift threshold" columns
showed "—" for a model explicitly reported as "Rejected — Failed drift
validation".
"""

from app.services.result_service import ResultService


def _service() -> ResultService:
    return ResultService(executor=object(), dataset_preview_service=object())


def _rejected_entry(model_name: str, statistic: float, threshold: float) -> dict:
    """Shaped exactly like RejectedCandidate.to_dict() in
    forecast_engine/s10_selection/selection_report.py."""
    return {
        "model_name": model_name,
        "reason": f"population_stability_index statistic {statistic:.6f} > threshold {threshold:.6f} (percentile).",
        "selected_drift_algorithm": "population_stability_index",
        "drift_statistic": statistic,
        "dynamic_threshold_method": "percentile",
        "dynamic_threshold_value": threshold,
    }


def test_a_rejected_models_drift_statistic_and_threshold_are_shown_not_dashed():
    rejected_reasons = {"xgboost": _rejected_entry("xgboost", 0.183245, 0.15)}

    detail = _service()._drift_detail(
        name="xgboost",
        winning_name="lightgbm",
        is_fallback_winner=False,
        drift={},
        rejected_reasons=rejected_reasons,
    )

    assert detail.evaluated is True
    assert detail.passed is False
    assert detail.algorithm == "population_stability_index"
    assert detail.statistic == 0.183245
    assert detail.threshold_value == 0.15
    assert detail.threshold_method == "percentile"


def test_each_rejected_model_shows_its_own_numbers_not_a_shared_one():
    rejected_reasons = {
        "xgboost": _rejected_entry("xgboost", 0.183245, 0.15),
        "arima": _rejected_entry("arima", 0.09, 0.12),
    }
    service = _service()

    xgboost_detail = service._drift_detail("xgboost", "lightgbm", False, {}, rejected_reasons)
    arima_detail = service._drift_detail("arima", "lightgbm", False, {}, rejected_reasons)

    assert xgboost_detail.statistic == 0.183245
    assert arima_detail.statistic == 0.09
    assert xgboost_detail.statistic != arima_detail.statistic


def test_a_candidate_rejected_for_having_no_forecast_is_not_evaluated():
    """No forward forecast means drift validation never ran for this
    candidate - evaluated must be False, and the numbers must be None, not
    a false zero or a copy of some other candidate's numbers."""
    rejected_reasons = {
        "xgboost": {
            "model_name": "xgboost",
            "reason": "No forward forecast is available to validate.",
            "selected_drift_algorithm": None,
            "drift_statistic": None,
            "dynamic_threshold_method": None,
            "dynamic_threshold_value": None,
        }
    }

    detail = _service()._drift_detail("xgboost", "lightgbm", False, {}, rejected_reasons)

    assert detail.evaluated is False
    assert detail.statistic is None
    assert detail.threshold_value is None
    assert detail.detail == "No forward forecast is available to validate."


def test_selection_outcome_still_reads_the_reason_text_from_the_richer_entry():
    """_selection_outcome's 'Rejected — <reason>' string must still read
    correctly now that rejected_reasons values are full dicts, not bare
    strings."""
    rejected_reasons = {"xgboost": _rejected_entry("xgboost", 0.183245, 0.15)}

    outcome = _service()._selection_outcome(
        name="xgboost",
        winning_name="lightgbm",
        is_fallback_winner=False,
        training={},
        evaluation={},
        rejected_reasons=rejected_reasons,
        ranking_entry=None,
    )

    assert outcome.startswith("Rejected — population_stability_index statistic 0.183245")


def test_the_winning_models_own_drift_detail_is_unaffected():
    """The winner's DriftDetail is built from the separate `drift` dict
    (result.drift_results), not from rejected_reasons - this fix must not
    touch that path."""
    drift = {
        "algorithm": "kolmogorov_smirnov",
        "statistic": 0.05,
        "threshold_value": 0.2,
        "threshold_method": "auto",
        "result": {"passed": True, "detail": "ok"},
    }

    detail = _service()._drift_detail("lightgbm", "lightgbm", False, drift, {})

    assert detail.evaluated is True
    assert detail.passed is True
    assert detail.statistic == 0.05
    assert detail.threshold_value == 0.2
