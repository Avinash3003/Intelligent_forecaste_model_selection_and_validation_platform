"""Rolls sub-monthly data up to the month grain.

Forecasting runs at month level, so Daily/Weekly data is resampled here.
Monthly passes through untouched; anything coarser is rejected by validation
rather than disaggregated.

The target's method is the user's choice (Sum/Mean/Last) — only they know
whether it is a flow to total or a level to average. Feature columns use a
configurable default instead, since summing a price column would be
meaningless, with per-column overrides available.
"""

from __future__ import annotations

import pandas as pd

from forecast_engine.config.pipeline_config import AggregationConfig
from forecast_engine.core.forecast_configuration import AggregationMethod, ForecastConfiguration

# Grains finer than monthly are rolled up; anything else is left alone.
SUB_MONTHLY_FREQUENCIES: frozenset[str] = frozenset({"Daily", "Weekly"})


class MonthlyAggregator:
    """Resamples a sub-monthly dataset to one row per key per month."""

    # Store aggregation configuration
    def __init__(self, config: AggregationConfig | None = None) -> None:
        self._config = config or AggregationConfig()

    # Whether frequency needs rolling up to month level
    def should_aggregate(self, frequency: str | None) -> bool:
        return bool(self._config.enabled) and frequency in SUB_MONTHLY_FREQUENCIES

    # Roll a dataframe up to the monthly grain
    def aggregate(
        self,
        dataframe: pd.DataFrame,
        configuration: ForecastConfiguration,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        date_column = configuration.date_column
        target_column = configuration.target_column
        key_columns = list(configuration.key_columns)

        working = dataframe.copy()
        rows_before = len(working)

        # Collapse each timestamp onto its month start; grouping on that
        # keeps the result a real datetime column rather than a period.
        month_start = working[date_column].dt.to_period("M").dt.to_timestamp()
        working[date_column] = month_start

        aggregation_map = self._build_aggregation_map(working, configuration)

        grouped = working.groupby([*key_columns, date_column], as_index=False, sort=True)
        aggregated = grouped.agg(aggregation_map)

        metadata = {
            "aggregation_applied": True,
            "aggregation_method": configuration.aggregation_method.value,
            "rows_before_aggregation": rows_before,
            "rows_after_aggregation": len(aggregated),
            "feature_aggregation": {
                column: method
                for column, method in aggregation_map.items()
                if column != target_column
            },
        }
        return aggregated, metadata

    # Decide the aggregation method for each non-key column
    def _build_aggregation_map(
        self, dataframe: pd.DataFrame, configuration: ForecastConfiguration
    ) -> dict[str, str]:
        # The target follows the user's selection. Features fall back to the
        # configured default, overridable per column, and non-numeric features
        # take the last value in the month since averaging them is undefined.
        aggregation_map: dict[str, str] = {
            configuration.target_column: self._resolve_target_method(configuration.aggregation_method)
        }

        for column in configuration.feature_columns:
            if column not in dataframe.columns:
                continue

            override = self._config.feature_methods.get(column)
            if override:
                aggregation_map[column] = override
            elif pd.api.types.is_numeric_dtype(dataframe[column]):
                aggregation_map[column] = self._config.default_feature_method
            else:
                aggregation_map[column] = "last"

        return aggregation_map

    # Translate the selected method into a pandas aggregation name
    def _resolve_target_method(self, method: AggregationMethod) -> str:
        return {
            AggregationMethod.SUM: "sum",
            AggregationMethod.MEAN: "mean",
            AggregationMethod.LAST: "last",
        }[method]
