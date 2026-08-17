"""The Actual vs Forecast payload must carry the COMPLETE history, a real
boundary, and real forward dates.

The chart's job is to plot every observation the run had for the selected
business key. These tests pin the payload the chart is built from — the
history is never truncated here, the boundary is the actual final
historical timestamp rather than a counted position, and forecast points
carry projected calendar labels so a long series does not switch from
dates to opaque T-keys halfway across.
"""

from types import SimpleNamespace

from app.orchestration.schemas import JobStatus
from app.services.result_service import ResultService, _projected_labels


def _months(count: int, start_year: int = 2013) -> list[str]:
    dates = []
    year, month = start_year, 1
    for _ in range(count):
        dates.append(f"{year:04d}-{month:02d}-01")
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return dates


class _FakePreview:
    """Stands in for the curated-file reader the chart's history comes from."""

    def __init__(self, series=None):
        self._series = series

    def get_full_series(self, run_id, date_column, target_column, key_values):
        return self._series


def _result(history_length: int = 60):
    return SimpleNamespace(
        run_id="run-1",
        job_status=JobStatus.COMPLETED,
        forecast_groups=[
            {
                "group_id": "1 | 1",
                "key_values": {"store": 1, "item": 1},
                # Deliberately far shorter than the curated history: this is
                # the bounded tail the summary carries, and the chart must
                # NOT be built from it when curated data exists.
                "recent_history": [
                    {"date": date, "value": 10.0} for date in _months(24)
                ],
            }
        ],
        run_metadata={"configuration": {"date_column": "date", "target_column": "sales"}},
    )


def _service(series):
    return ResultService(executor=object(), dataset_preview_service=_FakePreview(series))


def _winner(horizon: int = 5, with_interval: bool = True):
    forecast = {"values": [100.0 + i for i in range(horizon)]}
    if with_interval:
        forecast["lower"] = [90.0 + i for i in range(horizon)]
        forecast["upper"] = [110.0 + i for i in range(horizon)]
    return {"model_name": "arima", "forecast": forecast}


# ---------------------------------------------------------------------
# Full history — never a fixed window
# ---------------------------------------------------------------------


def test_every_curated_observation_is_plotted():
    dates = _months(60)
    series = [(date, float(index)) for index, date in enumerate(dates)]

    points = _service(series)._actual_vs_forecast(_result(), "1 | 1", _winner())

    actuals = [point for point in points if point.actual is not None]
    assert len(actuals) == 60, "history must not be truncated to a fixed window"
    assert actuals[0].period == dates[0]
    assert actuals[-1].period == dates[-1]


def test_a_very_long_series_is_not_thinned_server_side():
    """Label thinning is the chart's job; the payload stays complete."""
    dates = _months(360)  # 30 years of months
    series = [(date, float(index)) for index, date in enumerate(dates)]

    points = _service(series)._actual_vs_forecast(_result(), "1 | 1", _winner())

    assert len([point for point in points if point.actual is not None]) == 360


def test_the_bounded_tail_is_used_only_when_curated_data_is_missing():
    points = _service(None)._actual_vs_forecast(_result(), "1 | 1", _winner())

    actuals = [point for point in points if point.actual is not None]
    assert len(actuals) == 24, "fallback to recent_history must still work"


def test_a_short_history_is_plotted_whole():
    dates = _months(7)
    series = [(date, float(index)) for index, date in enumerate(dates)]

    points = _service(series)._actual_vs_forecast(_result(), "1 | 1", _winner(horizon=3))

    assert len([point for point in points if point.actual is not None]) == 7


# ---------------------------------------------------------------------
# The history/forecast boundary
# ---------------------------------------------------------------------


def test_the_boundary_marks_the_final_historical_timestamp():
    dates = _months(48)
    series = [(date, float(index)) for index, date in enumerate(dates)]

    points = _service(series)._actual_vs_forecast(_result(), "1 | 1", _winner())

    boundaries = [point for point in points if point.boundary]
    assert len(boundaries) == 1
    assert boundaries[0].period == dates[-1]
    # It is the junction: the last actual, also carrying the forecast's
    # first value so the two lines connect.
    assert boundaries[0].actual is not None
    assert boundaries[0].forecast == boundaries[0].actual


def test_the_boundary_position_does_not_depend_on_series_length():
    for length in (7, 36, 120):
        dates = _months(length)
        series = [(date, float(index)) for index, date in enumerate(dates)]
        points = _service(series)._actual_vs_forecast(_result(), "1 | 1", _winner())
        assert next(point for point in points if point.boundary).period == dates[-1]


def test_no_boundary_is_marked_when_there_is_no_forecast():
    dates = _months(12)
    series = [(date, float(index)) for index, date in enumerate(dates)]

    points = _service(series)._actual_vs_forecast(_result(), "1 | 1", {"forecast": {}})

    assert not any(point.boundary for point in points)


# ---------------------------------------------------------------------
# Forward date labels
# ---------------------------------------------------------------------


def test_forecast_points_carry_projected_calendar_labels():
    dates = _months(36)  # ends 2015-12-01
    series = [(date, float(index)) for index, date in enumerate(dates)]

    points = _service(series)._actual_vs_forecast(_result(), "1 | 1", _winner(horizon=3))

    forecast_points = [point for point in points if point.actual is None]
    assert [point.period for point in forecast_points] == ["T1", "T2", "T3"]
    assert [point.label for point in forecast_points] == ["2016-01-01", "2016-02-01", "2016-03-01"]


def test_the_horizon_key_stays_the_point_identity():
    """The horizon selector matches on `period`; labels must not replace it."""
    dates = _months(24)
    series = [(date, float(index)) for index, date in enumerate(dates)]

    points = _service(series)._actual_vs_forecast(_result(), "1 | 1", _winner(horizon=4))

    assert [p.period for p in points if p.actual is None] == ["T1", "T2", "T3", "T4"]


def test_actual_points_are_labelled_with_their_own_date():
    dates = _months(10)
    series = [(date, float(index)) for index, date in enumerate(dates)]

    points = _service(series)._actual_vs_forecast(_result(), "1 | 1", _winner())

    for point in points:
        if point.actual is not None:
            assert point.label == point.period


def test_projected_labels_follow_a_daily_cadence_too():
    daily = ["2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04"]
    assert _projected_labels(daily, 2) == ["2020-03-05", "2020-03-06"]


def test_projected_labels_clamp_to_month_length():
    assert _projected_labels(["2020-12-31", "2021-01-31"], 1) == ["2021-02-28"]


def test_projected_labels_degrade_when_cadence_is_unknowable():
    assert _projected_labels(["2020-01-01"], 3) == [None, None, None]
    assert _projected_labels([], 2) == [None, None]


def test_an_unlabelled_forecast_point_still_has_an_identity():
    """One observation is too few to establish a cadence — the point still
    renders as T1 rather than disappearing."""
    series = [("2020-01-01", 5.0)]

    points = _service(series)._actual_vs_forecast(_result(), "1 | 1", _winner(horizon=2))

    forecast_points = [point for point in points if point.actual is None]
    assert [point.period for point in forecast_points] == ["T1", "T2"]
    assert all(point.label is None for point in forecast_points)


# ---------------------------------------------------------------------
# Prediction intervals — preserved for models that produce them, never
# fabricated for models that do not
# ---------------------------------------------------------------------


def test_the_interval_is_preserved_for_a_model_that_produces_one():
    dates = _months(24)
    series = [(date, float(index)) for index, date in enumerate(dates)]

    points = _service(series)._actual_vs_forecast(_result(), "1 | 1", _winner(with_interval=True))

    forecast_points = [point for point in points if point.actual is None]
    assert all(point.lower is not None and point.upper is not None for point in forecast_points)


def test_the_band_is_anchored_at_the_boundary_with_zero_width():
    dates = _months(24)
    series = [(date, float(index)) for index, date in enumerate(dates)]

    points = _service(series)._actual_vs_forecast(_result(), "1 | 1", _winner(with_interval=True))

    boundary = next(point for point in points if point.boundary)
    # The last actual is an observation, so its interval has zero width —
    # a fact about the data, not an invented bound.
    assert boundary.lower == boundary.upper == boundary.actual


def test_no_interval_is_fabricated_for_a_model_without_one():
    dates = _months(24)
    series = [(date, float(index)) for index, date in enumerate(dates)]

    points = _service(series)._actual_vs_forecast(
        _result(), "1 | 1", _winner(with_interval=False)
    )

    assert all(point.lower is None and point.upper is None for point in points)
    boundary = next(point for point in points if point.boundary)
    assert boundary.lower is None and boundary.upper is None
