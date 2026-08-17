"""One pass over an uploaded dataset, producing everything the estimator needs.

Runtime cannot be estimated from row and column counts alone: 900,000 rows
could be 500 keys of 60 months or 5 keys of daily history, and those are
completely different amounts of work. What actually drives the estimate is
periods *per business key*, because model eligibility (`min_observations`)
and the rolling backtest's fold count are both functions of it.

So some analysis is genuinely required — but only once. This module exists
because the estimator used to parse the date column three separate times
(frequency detection, then `_periods_by_group` from two different callers)
over the whole frame. Everything derivable from one parse is now derived
from one parse, and the result is passed around instead of recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.schemas.metadata import MetadataMapping
from app.services.frequency_detector import FrequencyDetector

# What a column-less analysis reports as its grain, matching
# FrequencyDetector's own vocabulary for "the spacing says nothing".
UNKNOWN_GRAIN = "Unknown"


@dataclass(frozen=True)
class DatasetAnalysis:
    """The workload characteristics of one upload under one column mapping.

    Every field is measured, never assumed. `periods_by_group` is the
    important one — the distinct calendar months each business key holds,
    which is what the curated dataset will contain after the engine's
    monthly aggregation, and therefore the right basis for per-key
    eligibility and fold counts.
    """

    rows: int
    columns: int
    date_grain: str
    periods_by_group: dict[tuple, int] = field(default_factory=dict)
    # None when the target column is absent or unreadable as numeric —
    # never a fabricated 0%.
    missingness_pct: float | None = None
    key_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)

    @property
    def unique_keys(self) -> int:
        """Forecast groups this run will produce. A dataset with no key
        columns is one series, not zero."""
        return len(self.periods_by_group) or 1

    @property
    def period_counts(self) -> list[int]:
        """Per-group history depth, in the order groups were found.

        Falls back to a single entry for the longest group so a dataset
        whose keys could not be grouped still estimates against real
        history rather than against nothing.
        """
        return list(self.periods_by_group.values()) or [self.history_length_periods]

    @property
    def history_length_periods(self) -> int:
        """The deepest single group — what min-observation gates are
        measured against."""
        return int(max(self.periods_by_group.values(), default=0))


class DatasetAnalyzer:
    """Measures a DataFrame's workload characteristics in a single pass."""

    def __init__(self, frequency_detector: FrequencyDetector | None = None) -> None:
        self._frequency_detector = frequency_detector or FrequencyDetector()

    def analyze(self, dataframe: pd.DataFrame, metadata: MetadataMapping) -> DatasetAnalysis:
        key_columns = [c for c in metadata.key_columns if c in dataframe.columns]
        feature_columns = [c for c in metadata.feature_columns if c in dataframe.columns]

        date_grain = UNKNOWN_GRAIN
        periods_by_group: dict[tuple, int] = {}

        if metadata.date_column in dataframe.columns:
            # The one and only parse of the date column. Both the grain and
            # the per-group month counts are derived from this Series.
            parsed = pd.to_datetime(dataframe[metadata.date_column], errors="coerce")
            date_grain = self._frequency_detector.detect_parsed(parsed)
            periods_by_group = self._periods_by_group(dataframe, parsed, key_columns)

        return DatasetAnalysis(
            rows=int(dataframe.shape[0]),
            columns=int(dataframe.shape[1]),
            date_grain=date_grain,
            periods_by_group=periods_by_group,
            missingness_pct=self._missingness(dataframe, metadata.target_column),
            key_columns=key_columns,
            feature_columns=feature_columns,
        )

    def _periods_by_group(
        self, dataframe: pd.DataFrame, parsed_dates: pd.Series, key_columns: list[str]
    ) -> dict[tuple, int]:
        """Distinct months per business key, from an already-parsed column."""
        periods = parsed_dates.dt.to_period("M")

        if not key_columns:
            valid = periods.dropna()
            return {(): int(valid.nunique())} if len(valid) else {}

        try:
            grouped = periods.groupby([dataframe[c] for c in key_columns]).nunique()
        except (TypeError, ValueError):
            # Unhashable or otherwise ungroupable key values. Reported as
            # "no groups" rather than raised: the estimate degrades to the
            # single-series path instead of failing the request.
            return {}

        return {
            (key if isinstance(key, tuple) else (key,)): int(count)
            for key, count in grouped.items()
        }

    def _missingness(self, dataframe: pd.DataFrame, target_column: str) -> float | None:
        if target_column not in dataframe.columns:
            return None
        target = pd.to_numeric(dataframe[target_column], errors="coerce")
        return round(float(target.isna().mean() * 100.0), 2)
