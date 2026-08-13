"""Data Quality Assessment — analyses a raw dataset and reports on it.

Strictly read-only. This module never mutates, cleans, converts or drops a
single value; it only measures. That separation matters for auditability:
the report describes the dataset exactly as the user uploaded it, so the
"before" picture stays truthful no matter what preprocessing later does.

Everything it measures is driven by the user's metadata selection — which
column is the date, the target, a key — so no dataset-specific assumption
is ever made.

Runs immediately after loading, and its verdict gates preprocessing: a
dataset judged NOT_SUITABLE never reaches the cleaning stage.
"""

from __future__ import annotations

import pandas as pd

from forecast_engine.config.pipeline_config import PANDAS_FREQUENCY_ALIASES, QualityConfig
from forecast_engine.core.forecast_configuration import ForecastConfiguration
from forecast_engine.s02_quality.quality_report import ColumnQuality, QualityReport, SuitabilityStatus


# Parse a raw column as dates, the same way `QualityAssessor.assess()` does
# for `configuration.date_column` — module-level (not a method) so a caller
# that only wants a date range, not a full quality assessment, can reuse
# the exact same parsing rule without constructing a ForecastConfiguration.
# Numeric columns are refused rather than parsed — pandas would read
# integers as nanosecond epochs and report a meaningless date range.
def parse_date_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return pd.Series(pd.NaT, index=series.index)
    return pd.to_datetime(series, errors="coerce")


# The observed timeline bounds of an already-parsed date series, as ISO
# strings — (None, None) when nothing in it parsed. Never guesses; a
# caller with no valid dates gets an honest "unavailable", not a fabricated
# range.
def date_range_from_parsed(parsed_dates: pd.Series) -> tuple[str | None, str | None]:
    valid = parsed_dates.dropna()
    if valid.empty:
        return None, None
    return valid.min().isoformat(), valid.max().isoformat()


class DataQualityAssessor:
    """Produces a QualityReport for a raw dataset. Never modifies data."""

    # Store quality assessment configuration
    def __init__(self, config: QualityConfig | None = None) -> None:
        self._config = config or QualityConfig()

    # Analyse a dataframe against the user's metadata selection
    def assess(
        self,
        dataframe: pd.DataFrame,
        configuration: ForecastConfiguration,
        frequency: str | None = None,
    ) -> QualityReport:
        configuration.validate_against_columns(list(dataframe.columns))

        report = QualityReport(
            total_rows=int(len(dataframe)),
            total_columns=int(dataframe.shape[1]),
            frequency=frequency,
            columns=[self._profile_column(dataframe[name]) for name in dataframe.columns],
        )

        report.duplicate_rows = int(dataframe.duplicated().sum())
        report.duplicate_timestamps = self._count_duplicate_timestamps(dataframe, configuration)

        parsed_dates = parse_date_column(dataframe[configuration.date_column])
        report.invalid_date_values = self._count_unconvertible(
            dataframe[configuration.date_column], parsed_dates
        )
        report.date_range_start, report.date_range_end = date_range_from_parsed(parsed_dates)

        numeric_target = pd.to_numeric(dataframe[configuration.target_column], errors="coerce")
        report.invalid_target_values = self._count_unconvertible(
            dataframe[configuration.target_column], numeric_target
        )
        report.total_observations = int(numeric_target.notna().sum())
        report.constant_target = bool(numeric_target.nunique(dropna=True) <= 1)

        report.distinct_business_keys = self._count_business_keys(dataframe, configuration)
        report.missing_timestamps = self._count_missing_timestamps(
            dataframe, configuration, parsed_dates, frequency, report.distinct_business_keys
        )

        self._evaluate_suitability(report, configuration)
        return report

    # ------------------------------------------------------------------
    # Individual measurements
    # ------------------------------------------------------------------

    # Describe one column's completeness and cardinality
    def _profile_column(self, series: pd.Series) -> ColumnQuality:
        return ColumnQuality(
            name=str(series.name),
            dtype=str(series.dtype),
            missing_values=int(series.isna().sum()),
            null_percentage=float(series.isna().mean() * 100),
            distinct_values=int(series.nunique(dropna=True)),
        )

    # Count rows that repeat a timestamp within the same business key
    def _count_duplicate_timestamps(
        self, dataframe: pd.DataFrame, configuration: ForecastConfiguration
    ) -> int:
        # A time series admits one observation per key per period, so these are
        # defects even when the rows differ elsewhere — which is exactly why a
        # plain duplicate-row check does not catch them.
        subset = [*configuration.key_columns, configuration.date_column]
        return int(dataframe.duplicated(subset=subset).sum())

    # Count values that were present but failed to convert
    def _count_unconvertible(self, original: pd.Series, converted: pd.Series) -> int:
        # Genuinely missing entries are excluded — they are reported
        # separately as missing values, not as invalid ones.
        return int((original.notna() & converted.isna()).sum())

    # Count distinct forecasting groups implied by the key columns
    def _count_business_keys(
        self, dataframe: pd.DataFrame, configuration: ForecastConfiguration
    ) -> int:
        if not configuration.key_columns:
            return 1
        return int(dataframe[list(configuration.key_columns)].drop_duplicates().shape[0])

    # Count gaps between each group's first and last observation
    def _count_missing_timestamps(
        self,
        dataframe: pd.DataFrame,
        configuration: ForecastConfiguration,
        parsed_dates: pd.Series,
        frequency: str | None,
        group_count: int,
    ) -> int | None:
        # Returns None — rather than 0 — whenever the answer is unknowable or
        # deliberately skipped, so a genuine "no gaps" is never confused with
        # "not measured". The scan is capped by config because rebuilding an
        # expected timeline is linear in group count.
        alias = PANDAS_FREQUENCY_ALIASES.get(frequency or "")
        if not self._config.detect_missing_timestamps or alias is None:
            return None

        if group_count > self._config.max_groups_for_timestamp_scan:
            return None

        # Work on a private frame so the caller's dataset is untouched.
        working = pd.DataFrame({"_date": parsed_dates}).dropna()
        if working.empty:
            return None

        if configuration.key_columns:
            for column in configuration.key_columns:
                working[column] = dataframe.loc[working.index, column]
            groups = working.groupby(list(configuration.key_columns))["_date"]
        else:
            # Single series — the whole dataset is one implicit group.
            return self._gap_count(working["_date"], alias)

        return int(sum(self._gap_count(dates, alias) for _, dates in groups))

    # Missing periods between one series' first and last observation
    def _gap_count(self, dates: pd.Series, alias: str) -> int:
        distinct = dates.drop_duplicates()
        if len(distinct) < 2:
            return 0

        expected = pd.date_range(start=distinct.min(), end=distinct.max(), freq=alias)
        return max(int(len(expected) - len(distinct)), 0)

    # ------------------------------------------------------------------
    # Overall verdict
    # ------------------------------------------------------------------

    # Set the report's overall verdict and the reasons behind it
    def _evaluate_suitability(
        self, report: QualityReport, configuration: ForecastConfiguration
    ) -> None:
        # Blocking findings are those preprocessing cannot repair: no usable
        # timeline, a target that is categorical or constant, or too little
        # history. Everything preprocessing *can* fix is a warning.
        blocking: list[str] = []
        warnings: list[str] = []

        if report.total_observations == 0:
            blocking.append(
                f"Target column '{configuration.target_column}' contains no numeric values."
            )
        elif report.total_observations < self._config.min_observations:
            blocking.append(
                f"Dataset contains insufficient historical observations "
                f"({report.total_observations:,} usable, minimum {self._config.min_observations})."
            )

        if report.constant_target and report.total_observations > 0:
            blocking.append(
                f"Target column '{configuration.target_column}' is constant and cannot be forecast."
            )

        if report.date_range_start is None:
            blocking.append(
                f"Date column '{configuration.date_column}' contains no interpretable dates."
            )

        # A missing-target ratio beyond the configured ceiling is blocking:
        # dropping that many rows would leave too little history to trust.
        if report.total_rows > 0:
            missing_ratio = report.invalid_target_values / report.total_rows
            missing_ratio += self._target_null_ratio(report, configuration)
            if missing_ratio > self._config.max_target_missing_ratio:
                blocking.append(
                    f"Target column '{configuration.target_column}' is missing or unusable in "
                    f"{missing_ratio:.1%} of rows, above the {self._config.max_target_missing_ratio:.0%} limit."
                )
            elif missing_ratio > 0:
                warnings.append(
                    f"{missing_ratio:.1%} of target values are missing or unusable and will be removed."
                )

        if report.duplicate_rows:
            warnings.append(f"{report.duplicate_rows:,} duplicate row(s) will be removed.")

        if report.duplicate_timestamps:
            warnings.append(
                f"{report.duplicate_timestamps:,} row(s) repeat a timestamp within the same business key."
            )

        if report.invalid_date_values:
            warnings.append(
                f"{report.invalid_date_values:,} row(s) hold an unreadable date and will be removed."
            )

        if report.missing_timestamps:
            warnings.append(
                f"{report.missing_timestamps:,} expected period(s) are absent from the timeline."
            )

        if blocking:
            report.suitability = SuitabilityStatus.NOT_SUITABLE
            report.suitability_reasons = blocking
        elif warnings:
            report.suitability = SuitabilityStatus.WARNINGS
            report.suitability_reasons = warnings
        else:
            report.suitability = SuitabilityStatus.READY
            report.suitability_reasons = []

    # Share of target values that were absent (as opposed to invalid)
    def _target_null_ratio(
        self, report: QualityReport, configuration: ForecastConfiguration
    ) -> float:
        for column in report.columns:
            if column.name == configuration.target_column:
                return column.null_percentage / 100
        return 0.0
