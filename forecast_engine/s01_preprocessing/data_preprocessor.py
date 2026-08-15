"""Turns the raw dataset into the curated one.

The only module allowed to modify data — everything else reads or reshapes
— so this is the single place to look when a curated row count differs from
the raw one.

Where quality assessment describes problems, this acts on them:
  - date and target are converted or the dataset is rejected; silently
    coercing a categorical target to NaN would hide the problem until a
    model produced nonsense.
  - rows that cannot carry an observation are removed and counted.
  - key values and feature columns are left exactly as uploaded.

The input DataFrame is never mutated, so the caller keeps the raw data.
"""

from __future__ import annotations

import pandas as pd

from forecast_engine.config.pipeline_config import (
    AggregationConfig,
    ConversionConfig,
    DuplicateScope,
    MissingValueStrategy,
    PreprocessingConfig,
    QualityConfig,
)
from forecast_engine.core.forecast_configuration import ForecastConfiguration
from forecast_engine.s01_preprocessing.monthly_aggregator import MonthlyAggregator
from forecast_engine.s02_quality.quality_report import PreprocessingSummary
from forecast_engine.utils.exceptions import DataQualityError, PreprocessingError


class DataPreprocessor:
    """Produces a clean, type-standardized curated dataset."""

    # Store preprocessing, conversion, quality and aggregation config
    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        conversion: ConversionConfig | None = None,
        quality: QualityConfig | None = None,
        aggregation: AggregationConfig | None = None,
    ) -> None:
        self._config = config or PreprocessingConfig()
        self._conversion = conversion or ConversionConfig()
        self._quality = quality or QualityConfig()
        self._aggregator = MonthlyAggregator(aggregation or AggregationConfig())

    # Clean a dataframe into the curated dataset
    def prepare(
        self,
        dataframe: pd.DataFrame,
        configuration: ForecastConfiguration,
        frequency: str | None = None,
    ) -> tuple[pd.DataFrame, PreprocessingSummary]:
        configuration.validate_against_columns(list(dataframe.columns))

        # Copy first — the raw dataset must survive this stage untouched.
        curated = dataframe.copy()
        summary = PreprocessingSummary(rows_read=len(curated))
        summary.forecast_mode = configuration.mode.value

        curated, summary.date_converted, summary.invalid_date_rows_removed = self._standardize_dates(
            curated, configuration
        )
        curated, summary.target_converted = self._standardize_target(curated, configuration)
        curated, summary.duplicate_rows_removed = self._remove_duplicates(curated, configuration)
        curated, summary.missing_target_rows_removed = self._handle_missing_values(curated, configuration)

        # Roll up to monthly only after cleaning, so duplicates and unusable
        # rows can never be folded into a monthly total.
        curated = self._aggregate_to_monthly(curated, configuration, frequency, summary)

        curated = self._sort(curated, configuration)

        summary.curated_rows = len(curated)
        summary.rows_removed = summary.rows_read - summary.curated_rows

        self._assert_usable(curated, configuration, summary)

        summary.forecast_ready = True
        summary.status = "Completed"
        return curated, summary

    # ------------------------------------------------------------------
    # Type standardization
    # ------------------------------------------------------------------

    # Ensure the date column is real datetime, or reject the dataset
    def _standardize_dates(
        self, dataframe: pd.DataFrame, configuration: ForecastConfiguration
    ) -> tuple[pd.DataFrame, bool, int]:
        # Returns (frame, converted, rows_removed). `converted` is False when
        # the column already held datetimes and nothing had to change.
        column = configuration.date_column
        series = dataframe[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            return dataframe, False, 0

        if not self._conversion.convert_date_column:
            raise PreprocessingError(
                f"Date column '{column}' is not a datetime column and date conversion is disabled."
            )

        # Numeric columns are refused outright: pandas would read integers as
        # nanosecond epochs and yield plausible-looking nonsense.
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise PreprocessingError(
                f"Invalid date column — '{column}' holds numeric values that cannot be "
                f"interpreted as dates."
            )

        parsed = pd.to_datetime(series, errors="coerce", dayfirst=self._conversion.dayfirst)

        present = int(series.notna().sum())
        if present == 0:
            raise PreprocessingError(f"Invalid date column — '{column}' contains no values.")

        success_ratio = int(parsed.notna().sum()) / present
        if success_ratio < self._conversion.minimum_success_ratio:
            raise PreprocessingError(
                f"Invalid date column — only {success_ratio:.1%} of values in '{column}' could be "
                f"interpreted as dates."
            )

        dataframe[column] = parsed

        # The few rows that failed to parse have no place on the timeline.
        removed = 0
        if self._config.drop_unparseable_dates:
            before = len(dataframe)
            dataframe = dataframe[dataframe[column].notna()]
            removed = before - len(dataframe)

        return dataframe, True, removed

    # Ensure the target column is numeric, or reject the dataset
    def _standardize_target(
        self, dataframe: pd.DataFrame, configuration: ForecastConfiguration
    ) -> tuple[pd.DataFrame, bool]:
        # Rows whose target fails to convert become NaN here and are removed by
        # the missing-value step, which keeps "what was unusable" and "what to
        # do about it" as two separate, independently configurable decisions.
        column = configuration.target_column
        series = dataframe[column]

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            return dataframe, False

        if not self._conversion.convert_target_column:
            raise PreprocessingError(
                f"Target column '{column}' is not numeric and target conversion is disabled."
            )

        if pd.api.types.is_datetime64_any_dtype(series):
            raise PreprocessingError(
                f"Target column '{column}' holds datetime values and cannot be converted into "
                f"the numeric values required for forecasting."
            )

        converted = pd.to_numeric(series, errors="coerce")

        present = int(series.notna().sum())
        if present == 0:
            raise PreprocessingError(f"Target column '{column}' contains no values.")

        success_ratio = int(converted.notna().sum()) / present
        if success_ratio < self._conversion.minimum_success_ratio:
            raise PreprocessingError(
                f"Target column '{column}' cannot be converted into numeric values — only "
                f"{success_ratio:.1%} of its values are numeric. Categorical targets cannot be forecast."
            )

        dataframe[column] = converted
        return dataframe, True

    # ------------------------------------------------------------------
    # Row-level cleaning
    # ------------------------------------------------------------------

    # Drop duplicate rows according to the configured scope
    def _remove_duplicates(
        self, dataframe: pd.DataFrame, configuration: ForecastConfiguration
    ) -> tuple[pd.DataFrame, int]:
        # FULL_ROW removes only exact repeats. KEY_AND_DATE additionally
        # enforces one observation per key per period, keeping the last
        # occurrence — the convention for a corrected restatement.
        if not self._config.drop_duplicate_rows:
            return dataframe, 0

        before = len(dataframe)

        if self._config.duplicate_scope is DuplicateScope.KEY_AND_DATE:
            subset = [*configuration.key_columns, configuration.date_column]
            dataframe = dataframe.drop_duplicates(subset=subset, keep="last")
        else:
            dataframe = dataframe.drop_duplicates()

        return dataframe, before - len(dataframe)

    # Apply the configured strategy to missing target values
    def _handle_missing_values(
        self, dataframe: pd.DataFrame, configuration: ForecastConfiguration
    ) -> tuple[pd.DataFrame, int]:
        # Feature columns are intentionally left alone in this phase. Fill
        # strategies, where enabled, are applied per business key so one key's
        # values can never leak into another's.
        strategy = self._config.target_missing_value_strategy
        column = configuration.target_column
        key_columns = list(configuration.key_columns)

        before = len(dataframe)

        if strategy is MissingValueStrategy.NONE:
            return dataframe, 0

        if strategy is MissingValueStrategy.DROP:
            dataframe = dataframe.dropna(subset=[column])
            return dataframe, before - len(dataframe)

        grouped = dataframe.groupby(key_columns)[column] if key_columns else None

        if strategy is MissingValueStrategy.FORWARD_FILL:
            dataframe[column] = grouped.ffill() if grouped is not None else dataframe[column].ffill()
        elif strategy is MissingValueStrategy.BACKWARD_FILL:
            dataframe[column] = grouped.bfill() if grouped is not None else dataframe[column].bfill()
        elif strategy is MissingValueStrategy.INTERPOLATE:
            dataframe[column] = (
                grouped.transform(lambda values: values.interpolate())
                if grouped is not None
                else dataframe[column].interpolate()
            )
        elif strategy is MissingValueStrategy.ZERO:
            dataframe[column] = dataframe[column].fillna(0)
        elif strategy is MissingValueStrategy.MEAN:
            dataframe[column] = (
                grouped.transform(lambda values: values.fillna(values.mean()))
                if grouped is not None
                else dataframe[column].fillna(dataframe[column].mean())
            )
        elif strategy is MissingValueStrategy.MEDIAN:
            dataframe[column] = (
                grouped.transform(lambda values: values.fillna(values.median()))
                if grouped is not None
                else dataframe[column].fillna(dataframe[column].median())
            )

        # A fill can still leave leading gaps with nothing to carry forward.
        dataframe = dataframe.dropna(subset=[column])
        return dataframe, before - len(dataframe)

    # Roll a sub-monthly dataset up to the monthly grain
    def _aggregate_to_monthly(
        self,
        dataframe: pd.DataFrame,
        configuration: ForecastConfiguration,
        frequency: str | None,
        summary: PreprocessingSummary,
    ) -> pd.DataFrame:
        # A no-op for data already at (or coarser than) month level. The
        # outcome is recorded on the summary for internal traceability only —
        # no aggregation detail is surfaced to the frontend.
        if not self._aggregator.should_aggregate(frequency):
            summary.aggregation_applied = False
            return dataframe

        aggregated, metadata = self._aggregator.aggregate(dataframe, configuration)

        summary.aggregation_applied = True
        summary.aggregation_method = configuration.aggregation_method.value
        summary.rows_after_aggregation = len(aggregated)
        summary.aggregation_detail = metadata

        return aggregated

    # Sort by business key, then chronologically
    def _sort(self, dataframe: pd.DataFrame, configuration: ForecastConfiguration) -> pd.DataFrame:
        # Sorting by key first keeps each series' rows contiguous, which makes
        # per-group slicing cheap and gives every run a deterministic order.
        # Single-series datasets sort by date alone.
        sort_columns = [*configuration.key_columns, configuration.date_column]
        return dataframe.sort_values(by=sort_columns).reset_index(drop=True)

    # Reject a curated dataset that cleaning left unforecastable
    def _assert_usable(
        self,
        dataframe: pd.DataFrame,
        configuration: ForecastConfiguration,
        summary: PreprocessingSummary,
    ) -> None:
        if dataframe.empty:
            summary.status = "Failed"
            raise PreprocessingError(
                "Preprocessing removed every row. Check the selected date and target columns."
            )

        if len(dataframe) < self._quality.min_observations:
            summary.status = "Failed"
            raise DataQualityError(
                f"Dataset contains insufficient historical observations after cleaning "
                f"({len(dataframe):,} rows, minimum {self._quality.min_observations})."
            )

        # A target left constant by cleaning has nothing to learn from.
        if dataframe[configuration.target_column].nunique(dropna=True) <= 1:
            summary.status = "Failed"
            raise DataQualityError(
                f"Target column '{configuration.target_column}' is constant after cleaning and "
                f"cannot be forecast."
            )
