"""Priority B — the Forecast Pipeline wizard's Metadata Mapping step shows
the real date coverage for whichever column the user has tentatively
assigned as the date column, via `compute_date_range()`.
"""

from __future__ import annotations

import pandas as pd

from app.services.profile_service import compute_date_range


def test_returns_the_real_min_and_max_dates():
    series = pd.Series(["2021-01-01", "2023-06-15", "2025-02-28"])
    start, end = compute_date_range(series)
    assert start == "2021-01-01T00:00:00"
    assert end == "2025-02-28T00:00:00"


def test_unordered_values_still_resolve_to_the_true_min_and_max():
    series = pd.Series(["2024-06-01", "2021-01-01", "2022-03-01"])
    start, end = compute_date_range(series)
    assert start == "2021-01-01T00:00:00"
    assert end == "2024-06-01T00:00:00"


def test_missing_or_invalid_values_are_ignored_not_a_crash():
    series = pd.Series(["2021-01-01", None, "not a date", "2022-01-01"])
    start, end = compute_date_range(series)
    assert start == "2021-01-01T00:00:00"
    assert end == "2022-01-01T00:00:00"


def test_a_column_with_no_parseable_dates_is_reported_as_unavailable():
    series = pd.Series(["not a date", "also not a date", None])
    assert compute_date_range(series) == (None, None)


def test_a_numeric_column_is_refused_rather_than_read_as_epoch_nanoseconds():
    # An id/count column parsed as dates would otherwise report a
    # nonsensical range (pandas reads bare integers as epoch nanoseconds).
    series = pd.Series([1, 2, 3, 4])
    assert compute_date_range(series) == (None, None)


def test_an_already_datetime_column_works_directly():
    series = pd.to_datetime(pd.Series(["2021-01-01", "2021-06-01"]))
    start, end = compute_date_range(series)
    assert start == "2021-01-01T00:00:00"
    assert end == "2021-06-01T00:00:00"
