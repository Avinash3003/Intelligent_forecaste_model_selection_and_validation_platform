"""Turns groups into the forecast-ready series training consumes.

One chronologically sorted frame per business key, holding the timeline,
the target, and any exogenous regressors.

Projects and orders only, never alters values — which is what makes the
later forward-validation and drift stages meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from forecast_engine.config.pipeline_config import GroupingConfig
from forecast_engine.core.forecast_configuration import ForecastConfiguration
from forecast_engine.s01_preprocessing.group_generator import ForecastGroup


@dataclass
class ForecastSeries:
    """One business key's forecast-ready series.

    frame holds date, target and feature columns only, sorted chronologically.
    date_column/target_column are recorded so consumers address columns by
    role without re-reading the run configuration. frequency is carried for
    reference; the series stays at its original grain.
    """

    group_id: str
    frame: pd.DataFrame = field(repr=False)
    date_column: str
    target_column: str
    feature_columns: tuple[str, ...] = ()
    key_values: dict[str, Any] = field(default_factory=dict)
    frequency: str | None = None
    meets_minimum_history: bool = True

    # Number of observations in the series
    @property
    def observation_count(self) -> int:
        return len(self.frame)

    # First timestamp, or None for an empty series
    @property
    def start_date(self) -> datetime | None:
        return self.frame[self.date_column].min() if not self.frame.empty else None

    # Last timestamp, or None for an empty series
    @property
    def end_date(self) -> datetime | None:
        return self.frame[self.date_column].max() if not self.frame.empty else None

    # The last `points` observations, for actual-vs-forecast charting
    def recent_history(self, points: int = 24) -> list[dict[str, Any]]:
        # Bounded rather than the full series: the dashboard only plots a
        # trailing window, and an unbounded dump would make the run summary
        # scale with dataset size.
        if self.frame.empty:
            return []

        tail = self.frame.tail(points)
        return [
            {"date": pd.Timestamp(row[self.date_column]).isoformat(), "value": _finite(row[self.target_column])}
            for _, row in tail.iterrows()
        ]

    # Serializable description, plus a bounded history tail
    def to_dict(self) -> dict[str, Any]:
        return {
            "recent_history": self.recent_history(),
            "group_id": self.group_id,
            "key_values": self.key_values,
            "observation_count": self.observation_count,
            "start_date": self.start_date.isoformat() if self.start_date is not None else None,
            "end_date": self.end_date.isoformat() if self.end_date is not None else None,
            "frequency": self.frequency,
            "feature_columns": list(self.feature_columns),
            "meets_minimum_history": self.meets_minimum_history,
        }


class SeriesBuilder:
    """Builds ForecastSeries objects from ForecastGroups."""

    # Store grouping configuration
    def __init__(self, config: GroupingConfig | None = None) -> None:
        self._config = config or GroupingConfig()

    # Build one forecast-ready series per group
    def build(
        self,
        groups: list[ForecastGroup],
        configuration: ForecastConfiguration,
        frequency: str | None = None,
    ) -> list[ForecastSeries]:
        return [self._build_one(group, configuration, frequency) for group in groups]

    # Project one group down to its series columns and sort it
    def _build_one(
        self,
        group: ForecastGroup,
        configuration: ForecastConfiguration,
        frequency: str | None,
    ) -> ForecastSeries:
        # Only date/target/feature columns are kept: key columns are constant
        # within a group and are preserved on the series' `key_values`
        # instead of being repeated on every row.
        series_columns = [column for column in configuration.series_columns if column in group.frame.columns]

        frame = (
            group.frame[series_columns]
            .sort_values(by=configuration.date_column)
            .reset_index(drop=True)
        )

        return ForecastSeries(
            group_id=group.group_id,
            frame=frame,
            date_column=configuration.date_column,
            target_column=configuration.target_column,
            feature_columns=configuration.feature_columns,
            key_values=group.key_values,
            frequency=frequency,
            meets_minimum_history=len(frame) >= self._config.min_observations_per_group,
        )


# Coerce one target value to a JSON-safe float, or None if unusable
def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
