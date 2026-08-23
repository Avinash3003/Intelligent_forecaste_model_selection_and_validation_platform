"""Confidence's stability component is judged against the full series.

`recent_history` is a 24-point tail, bounded purely to keep the run summary
small enough to ship for charting. Scoring a 30%-weighted confidence
component against that tail made the number depend on a display constant:
on a long series the tail is a small, unrepresentative slice of the real
variation, so a model whose forecast happened to match the last two years
could outscore a genuinely more accurate one — surfacing as the reported
inversion where a model with *higher* WMAPE showed *higher* confidence.
"""

from __future__ import annotations

from app.services.confidence import compute_confidence


def _stable_kwargs(**overrides):
    base = dict(
        wmape=20.0,
        drift_statistic=None,
        drift_threshold=None,
        is_fallback=False,
    )
    base.update(overrides)
    return base


def test_stability_follows_the_history_it_is_given():
    """A forecast matching the full series' spread must not be penalised
    just because a short recent window happens to be much calmer."""
    # Ten years of a strongly seasonal series: wide overall spread.
    full_history = [100.0, 500.0] * 60
    # The trailing 24 points of a *calm* stretch — the shape `recent_history`
    # would have handed us on a series whose recent period is quiet.
    calm_tail = [300.0, 310.0] * 12

    forecast = [100.0, 500.0] * 6  # matches the real, full-series variation

    against_full = compute_confidence(
        **_stable_kwargs(), forecast_values=forecast, history_values=full_history
    )
    against_tail = compute_confidence(
        **_stable_kwargs(), forecast_values=forecast, history_values=calm_tail
    )

    # Same model, same forecast, same WMAPE — only the history window differs.
    assert against_full.backtest_accuracy == against_tail.backtest_accuracy
    # Judged against the series it was actually fitted on, this forecast is
    # stable; judged against a calm 24-point tail it looks wildly volatile.
    assert against_full.forecast_stability == 1.0
    assert against_tail.forecast_stability == 0.0
    assert against_full.confidence > against_tail.confidence


def test_confidence_is_not_accuracy_alone():
    """The blend is deliberate: a more accurate model can still score lower
    overall. This is the behaviour the UI must not label as bare 'WMAPE'."""
    history = [100.0, 200.0] * 30

    accurate_but_flat = compute_confidence(
        wmape=19.47,
        drift_statistic=None,
        drift_threshold=None,
        is_fallback=False,
        forecast_values=[150.0] * 12,          # near-flat: poor stability
        history_values=history,
    )
    less_accurate_but_faithful = compute_confidence(
        wmape=23.97,
        drift_statistic=None,
        drift_threshold=None,
        is_fallback=False,
        forecast_values=[100.0, 200.0] * 6,     # matches history's variation
        history_values=history,
    )

    assert accurate_but_flat.backtest_accuracy > less_accurate_but_faithful.backtest_accuracy
    assert less_accurate_but_faithful.confidence > accurate_but_flat.confidence
    # The formula names every component that contributed, so the apparent
    # inversion is explainable on screen rather than looking like a bug.
    assert "stability" in less_accurate_but_faithful.formula


def test_missing_component_is_renormalised_not_scored_zero():
    with_drift = compute_confidence(
        wmape=20.0,
        drift_statistic=0.1,
        drift_threshold=1.0,
        is_fallback=False,
        forecast_values=[100.0, 200.0] * 6,
        history_values=[100.0, 200.0] * 30,
    )
    without_drift = compute_confidence(
        wmape=20.0,
        drift_statistic=None,
        drift_threshold=None,
        is_fallback=True,
        forecast_values=[100.0, 200.0] * 6,
        history_values=[100.0, 200.0] * 30,
    )

    assert without_drift.drift_margin is None
    # An unmeasurable drift margin must not drag the score toward zero.
    assert without_drift.confidence > 0.8
    assert "drift margin not applicable" in without_drift.formula
    assert with_drift.confidence is not None
