"""_model_decision()'s confidence input now uses a recent reference window
for forecast-stability instead of the group's entire observed history —
the same fix applied on the engine side, for the same reason: on a series
spanning multiple price/volume regimes, comparing a forecast against
decades of a since-departed regime made a good, regime-continuing forecast
look "unstable" (traced to a structural 0% confidence on the fallback
path in the forensic diagnostic). confidence.py's own formula/weights are
untouched — only what result_service.py passes as history_values changes.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.orchestration.schemas import JobStatus
from app.services.result_service import ResultService


class _FakePreview:
    def __init__(self, series):
        self._series = series

    def get_full_series(self, run_id, date_column, target_column, key_values):
        return self._series


def _months(count: int, start_year: int = 2000) -> list[str]:
    dates = []
    year, month = start_year, 1
    for _ in range(count):
        dates.append(f"{year:04d}-{month:02d}-01")
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return dates


def _result(frequency: str = "Monthly"):
    return SimpleNamespace(
        run_id="run-1",
        job_status=JobStatus.COMPLETED,
        run_metadata={
            "configuration": {"date_column": "date", "target_column": "target"},
            "frequency": frequency,
        },
        forecast_groups=[{"group_id": "g1", "key_values": {}}],
        forecast_results={"results": []},
    )


def _service(series):
    return ResultService(executor=object(), dataset_preview_service=_FakePreview(series))


def _winner(forecast_values):
    return {"model_name": "arima", "fallback_used": False, "forecast": {"values": forecast_values}}


def test_stability_is_computed_against_the_recent_window_not_the_full_regime_spanning_history():
    """The core regression: a forecast continuing the CURRENT regime must
    not be penalised by comparison against a much lower, since-departed
    old regime that dominates the full series by row count."""
    old_regime = [(d, 20.0 + i % 3) for i, d in enumerate(_months(120))]
    new_regime = [(d, 100.0 + i % 5) for i, d in enumerate(_months(36, start_year=2010))]
    series = old_regime + new_regime

    forecast_matching_current_regime = [100.0 + (i % 5) for i in range(12)]

    decision = _service(series)._model_decision(
        _result(), "g1", _winner(forecast_matching_current_regime), drift={}
    )

    # A forecast that matches the CURRENT regime's own variation should
    # score well on stability once compared against the right window.
    assert decision.confidence_explanation.forecast_stability is not None
    assert decision.confidence_explanation.forecast_stability > 0.8


def test_genuinely_unstable_forecast_still_scores_low():
    """The window recalibrates what's normal - it does not launder a
    forecast that is actually flat relative to recent behaviour."""
    old_regime = [(d, 20.0 + i % 3) for i, d in enumerate(_months(120))]
    new_regime = [(d, 100.0 + i % 5) for i, d in enumerate(_months(36, start_year=2010))]
    series = old_regime + new_regime

    flat_forecast = [100.0] * 12  # no variation at all, unlike recent history

    decision = _service(series)._model_decision(_result(), "g1", _winner(flat_forecast), drift={})

    assert decision.confidence_explanation.forecast_stability is not None
    assert decision.confidence_explanation.forecast_stability < 0.5


def test_short_history_behaves_as_before_full_series_used():
    series = [(d, 50.0 + i) for i, d in enumerate(_months(10))]
    forecast = [60.0 + i for i in range(3)]

    decision = _service(series)._model_decision(_result(), "g1", _winner(forecast), drift={})

    # Too short to window meaningfully - degrades to using everything,
    # same as before this change.
    assert decision.confidence_explanation.forecast_stability is not None
