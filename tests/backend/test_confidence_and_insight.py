"""Confidence composition and per-group insight consumption.

Both cover defects found on a real run (`fe-run-1717e57927bd`, group
`1 | 4`): a fallback forecast that flatlined still reported 83% confidence,
and the insight panel showed a *different* group's decision.

The second defect is now structurally impossible rather than patched: the
engine generates one structured JSON payload per group (Section 6.1 Task
10) instead of one free-text narrative the backend had to slice apart, so
there is no parser left to have a bug in. These tests check the new
direct-mapping path (`ResultService._explainability`) reads the *requested*
group's own dict and never another group's.
"""

import pytest

from app.services.confidence import compute_confidence
from app.services.result_service import ResultService

# Group 1 | 4's real numbers.
FLAT_FORECAST = [498.5, 500.0, 501.5, 503.0, 504.5, 506.0, 507.5, 509.0, 510.5, 512.0, 513.5, 515.0]
SEASONAL_HISTORY = [488, 464, 622, 724, 742, 814, 917, 780, 649, 680, 701, 497]


# ---------------------------------------------------------------------
# Per-group structured insight consumption
# ---------------------------------------------------------------------


def _result(business_insights):
    class _Result:
        pass

    result = _Result()
    result.business_insights = business_insights
    return result


def _service():
    return ResultService.__new__(ResultService)  # no executor needed for _explainability alone


BUSINESS_INSIGHTS = {
    "available": True,
    "status": "generated",
    "prompt_version": "v2",
    "groups": {
        "1 | 1": {
            "insight": {
                "selected_model": "xgboost", "rejection_reasons": [], "confidence": 74.1,
                "caveats": [], "concise_summary": "XGBoost was selected with a WMAPE of 25.9%.",
            },
            "provider": "azure_openai", "validation_status": "passed", "grounding_status": "grounded",
        },
        "1 | 4": {
            "insight": {
                "selected_model": "lightgbm", "rejection_reasons": ["xgboost: drift statistic exceeded threshold"],
                "confidence": 52.0, "caveats": ["fallback model used"],
                "concise_summary": "LightGBM was used as a fallback because XGBoost failed drift validation.",
            },
            "provider": "template", "validation_status": "passed", "grounding_status": "grounded",
        },
        "1 | 5": {
            "insight": {
                "selected_model": "arima", "rejection_reasons": [], "confidence": 84.5,
                "caveats": [], "concise_summary": "ARIMA was selected with a WMAPE of 15.5%.",
            },
            "provider": "azure_openai", "validation_status": "passed", "grounding_status": "grounded",
        },
    },
}


def test_each_group_shows_only_its_own_decision():
    # The defect: a whole-run narrative meant every group could show group
    # 1 | 1's text. Each group now reads its own dict entry directly.
    service = _service()
    result = _result(BUSINESS_INSIGHTS)

    narrative_1_1 = service._explainability(result, False, "1 | 1")
    narrative_1_4 = service._explainability(result, True, "1 | 4")
    narrative_1_5 = service._explainability(result, False, "1 | 5")

    assert "XGBoost was selected" in narrative_1_1.insight.summary
    assert "LightGBM was used as a fallback" in narrative_1_4.insight.summary
    assert "ARIMA was selected" in narrative_1_5.insight.summary


def test_a_group_the_run_never_produced_an_insight_for_shows_unavailable():
    # Showing another group's decision is worse than showing none.
    service = _service()
    narrative = service._explainability(_result(BUSINESS_INSIGHTS), False, "2 | 1")
    assert narrative.available is False
    assert narrative.paragraphs == []


def test_fallback_group_carries_its_own_caveat_and_rejection_reason():
    service = _service()
    narrative = service._explainability(_result(BUSINESS_INSIGHTS), True, "1 | 4")

    assert narrative.insight.caveat == "fallback model used"
    assert narrative.insight.key_reason == "Fallback model used"
    assert "Rejected candidates" in narrative.paragraphs[0]


def test_key_reason_reflects_rejection_count_when_not_a_fallback():
    insights = {
        "groups": {
            "g": {
                "insight": {
                    "selected_model": "xgboost",
                    "rejection_reasons": ["prophet: x", "arima: y"],
                    "confidence": 80.0, "caveats": [], "concise_summary": "XGBoost was selected over two rivals.",
                },
            }
        }
    }
    service = _service()
    narrative = service._explainability(_result(insights), False, "g")
    assert narrative.insight.key_reason == "Outranked 2 rejected candidates"


def test_no_business_insights_at_all_is_handled_cleanly():
    service = _service()
    narrative = service._explainability(_result({}), False, "1 | 1")
    assert narrative.available is False
    assert narrative.insight.summary is None


def test_full_explanation_never_repeats_the_card_summary():
    """The defect: "Full explanation" expanded to the exact sentence already
    shown in the card body, because `paragraphs[0]` was the same
    `concise_summary` string as `insight.summary`. `paragraphs` must only
    ever carry content the compact card doesn't already show.
    """
    service = _service()

    # A group with no rejection reasons and no caveats: nothing new to add,
    # so there is nothing to expand into — never a restated summary.
    narrative = service._explainability(_result(BUSINESS_INSIGHTS), False, "1 | 1")
    assert narrative.paragraphs == []

    # A group with a rejection reason: paragraphs carries that reason, never
    # the summary text itself.
    narrative_fallback = service._explainability(_result(BUSINESS_INSIGHTS), True, "1 | 4")
    assert narrative_fallback.insight.summary not in narrative_fallback.paragraphs
    assert all("was used as a fallback" not in p for p in narrative_fallback.paragraphs)


def test_only_caveats_beyond_the_first_appear_in_paragraphs():
    # caveats[0] is already shown as `insight.caveat` on the card; only the
    # rest belong in the expanded view.
    insights = {
        "groups": {
            "g": {
                "insight": {
                    "selected_model": "xgboost",
                    "rejection_reasons": [],
                    "confidence": 80.0,
                    "caveats": ["short history", "high variance", "one outlier period"],
                    "concise_summary": "XGBoost was selected.",
                },
            }
        }
    }
    service = _service()
    narrative = service._explainability(_result(insights), False, "g")

    assert narrative.insight.caveat == "short history"
    assert len(narrative.paragraphs) == 1
    assert "high variance" in narrative.paragraphs[0]
    assert "one outlier period" in narrative.paragraphs[0]
    assert "short history" not in narrative.paragraphs[0]


# ---------------------------------------------------------------------
# Confidence composition
# ---------------------------------------------------------------------


def test_flat_fallback_forecast_is_penalised():
    """The defect this exists for.

    Backtest accuracy alone read 83% for a forecast whose variation was 4%
    of its own history's — accuracy measured on history cannot see a flat
    forward line, which is exactly what the stability component is for.
    """
    result = compute_confidence(
        wmape=17.34,
        drift_statistic=None,
        drift_threshold=None,
        is_fallback=True,
        forecast_values=FLAT_FORECAST,
        history_values=SEASONAL_HISTORY,
    )

    assert result.forecast_stability == 0.0
    assert result.backtest_accuracy == pytest.approx(0.8266, abs=1e-3)
    # Materially below the 83% it used to report, without collapsing to zero
    # — the backtest evidence is real, it is just not the whole story.
    assert 0.4 < result.confidence < 0.6
    assert "stability" in result.formula


def test_a_forecast_matching_its_history_scores_high_stability():
    history = [100, 140, 90, 150, 95, 145, 105, 135, 92, 148, 98, 142]
    forecast = [102, 138, 93, 147, 97, 143, 103, 137, 94, 146, 96, 144]
    result = compute_confidence(
        wmape=10.0,
        drift_statistic=0.2,
        drift_threshold=0.8,
        is_fallback=False,
        forecast_values=forecast,
        history_values=history,
    )
    assert result.forecast_stability > 0.8


def test_excessive_volatility_is_penalised_like_flatness():
    """Section 6.5.1 and 6.5.2 are both failures of the same measure."""
    history = [100, 105, 98, 102, 101, 99, 103, 97, 104, 100, 102, 98]
    wild = [10, 400, 5, 500, 2, 600, 1, 700, 3, 550, 8, 480]
    flat = [100.0] * 12

    wild_result = compute_confidence(
        wmape=10.0, drift_statistic=None, drift_threshold=None, is_fallback=False,
        forecast_values=wild, history_values=history,
    )
    flat_result = compute_confidence(
        wmape=10.0, drift_statistic=None, drift_threshold=None, is_fallback=False,
        forecast_values=flat, history_values=history,
    )
    assert wild_result.forecast_stability == 0.0
    assert flat_result.forecast_stability == 0.0


def test_all_three_components_are_used_when_available():
    result = compute_confidence(
        wmape=20.0,
        drift_statistic=0.4,
        drift_threshold=0.8,
        is_fallback=False,
        forecast_values=[100, 120, 95, 125, 98, 122, 101, 118, 97, 124, 99, 121],
        history_values=[100, 130, 90, 135, 95, 128, 102, 125, 92, 132, 96, 127],
    )
    assert result.backtest_accuracy is not None
    assert result.forecast_stability is not None
    assert result.drift_margin is not None
    for label in ("backtest accuracy", "forecast stability", "drift margin"):
        assert label in result.formula


def test_missing_stability_is_renormalized_not_scored_zero():
    """An unmeasurable component must not be evidence of a bad forecast."""
    with_history = compute_confidence(
        wmape=20.0, drift_statistic=0.4, drift_threshold=0.8, is_fallback=False,
        forecast_values=[100, 120, 95, 125, 98, 122, 101, 118, 97, 124, 99, 121],
        history_values=[100, 130, 90, 135, 95, 128, 102, 125, 92, 132, 96, 127],
    )
    without = compute_confidence(
        wmape=20.0, drift_statistic=0.4, drift_threshold=0.8, is_fallback=False,
        forecast_values=None, history_values=None,
    )
    assert without.forecast_stability is None
    # Weights renormalize onto the two components that exist, so dropping an
    # unmeasurable one does not drag the score down.
    assert without.confidence > with_history.confidence


def test_constant_history_does_not_score_stability():
    # A flat forecast is the correct answer for a flat series; scoring it
    # as unstable would penalise the right behaviour.
    result = compute_confidence(
        wmape=5.0, drift_statistic=None, drift_threshold=None, is_fallback=False,
        forecast_values=[50.0] * 12, history_values=[50.0] * 12,
    )
    assert result.forecast_stability is None


def test_no_evidence_at_all_yields_no_number():
    result = compute_confidence(
        wmape=None, drift_statistic=None, drift_threshold=None, is_fallback=True,
        forecast_values=None, history_values=None,
    )
    assert result.confidence is None
    assert result.formula == "Not available"
