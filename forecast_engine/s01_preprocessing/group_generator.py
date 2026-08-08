"""Group Generator — splits a prepared dataset into forecasting groups.

One group == one business key == one thing that will get its own model
selection, ranking and drift validation downstream. This is where the
platform's core premise takes effect: because every key has different
statistical behaviour, a single model across all keys under-performs a
per-key choice (Section 2).

The split is driven entirely by the configured key columns:

  * No key columns  -> exactly one group covering the whole dataset
                       (single-series mode).
  * Key columns set -> one group per distinct combination of their values
                       (multi-series mode), e.g. (Store1, Item1),
                       (Store1, Item2), ...

Nothing here knows what those columns mean; "Store"/"Item" never appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from forecast_engine.config.pipeline_config import GroupingConfig
from forecast_engine.core.forecast_configuration import ForecastConfiguration
from forecast_engine.utils.exceptions import GroupGenerationError


# Convert a pandas/numpy scalar to its native Python equivalent
def _to_python_scalar(value: Any) -> Any:
    # groupby yields numpy types (np.int64, np.bool_) which are not JSON
    # serializable. Normalizing here — at the one point key values enter the
    # pipeline — keeps every downstream `to_dict()` safe to log or return
    # over an API without each caller re-handling the conversion.
    # numpy scalars expose .item(); native Python types do not.
    return value.item() if hasattr(value, "item") else value


@dataclass
class ForecastGroup:
    """One business key's slice of the prepared dataset.

    Attributes:
        group_id: Stable, readable identifier for the key, e.g. "1 | 3".
        key_values: The key columns and their values for this group.
            Empty in single-series mode.
        frame: The rows belonging to this key, already sorted.
    """

    group_id: str
    key_values: dict[str, Any] = field(default_factory=dict)
    frame: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    # Number of rows (observations) available for this key
    @property
    def observation_count(self) -> int:
        return len(self.frame)

    # Serializable description, excluding the data itself
    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "key_values": self.key_values,
            "observation_count": self.observation_count,
        }


class GroupGenerator:
    """Generates one ForecastGroup per business key."""

    # Store grouping configuration
    def __init__(self, config: GroupingConfig | None = None) -> None:
        self._config = config or GroupingConfig()

    # Split a dataframe into forecasting groups
    def generate(
        self, dataframe: pd.DataFrame, configuration: ForecastConfiguration
    ) -> list[ForecastGroup]:
        if dataframe.empty:
            raise GroupGenerationError("Cannot generate forecasting groups from an empty dataset.")

        groups = (
            self._generate_multi_series(dataframe, configuration)
            if configuration.key_columns
            else self._generate_single_series(dataframe)
        )

        if not groups:
            raise GroupGenerationError(
                "No forecasting groups could be generated from the dataset using key column(s) "
                f"{', '.join(configuration.key_columns)}."
            )

        # Thin keys can still be forecast, just with lower confidence, so
        # the default is to flag rather than discard them (Section 10).
        if self._config.drop_groups_below_minimum:
            groups = [
                group
                for group in groups
                if group.observation_count >= self._config.min_observations_per_group
            ]
            if not groups:
                raise GroupGenerationError(
                    "Every forecasting group fell below the configured minimum of "
                    f"{self._config.min_observations_per_group} observations."
                )

        return groups

    # Treat the whole dataset as one series
    def _generate_single_series(self, dataframe: pd.DataFrame) -> list[ForecastGroup]:
        # Used when no key columns are selected — the dataset already
        # represents a single entity, so there is nothing to split on.
        return [
            ForecastGroup(
                group_id=self._config.single_series_group_id,
                key_values={},
                frame=dataframe.copy(),
            )
        ]

    # Produce one group per distinct key-column combination
    def _generate_multi_series(
        self, dataframe: pd.DataFrame, configuration: ForecastConfiguration
    ) -> list[ForecastGroup]:
        # Rows missing any key value are excluded by pandas' groupby, which
        # is the behaviour we want: a row that cannot be attributed to a
        # specific business key cannot belong to any series.
        key_columns = list(configuration.key_columns)
        groups: list[ForecastGroup] = []

        for key_tuple, frame in dataframe.groupby(key_columns, sort=True, dropna=True):
            # groupby yields a scalar for a single key column and a tuple
            # for several; normalize so key_values is built the same way.
            values = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
            key_values = {column: _to_python_scalar(value) for column, value in zip(key_columns, values)}

            groups.append(
                ForecastGroup(
                    group_id=self._build_group_id(values),
                    key_values=key_values,
                    frame=frame.reset_index(drop=True),
                )
            )

        return groups

    # Join key values into one readable, stable identifier
    def _build_group_id(self, values: tuple[Any, ...]) -> str:
        # The id is what a user sees on the dashboard and what run artifacts
        # are keyed by, so it uses the configured separator rather than a
        # hash — traceability beats compactness here.
        return self._config.key_separator.join(str(value) for value in values)
