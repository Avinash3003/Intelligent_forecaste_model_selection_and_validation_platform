"""Decides whether the user's column choices can actually be forecast.

Every forecasting judgement lives here — the frontend makes none, it just
renders this report — so rules can change without a UI release.

Two things shape the implementation:
  - Three states, not two. CSVs store almost everything as text, so a column
    holding "2025-09-09" is CONVERTIBLE (safely cast during preprocessing),
    not INVALID. Only genuinely uncastable values ("Apple") are INVALID.
  - Checks read a bounded sample rather than converting whole columns, so
    validation stays interactive on large uploads.

Nothing here names a specific column; every check reads the assigned roles.
"""

import pandas as pd

from app.schemas.metadata import (
    ForecastConfigurationSummary,
    ForecastSuitability,
    MetadataValidationResponse,
    NormalizedMetadataConfig,
    SuitabilityStatus,
    ValidationCheckItem,
    ValidationStatus,
)

# Number of values inspected when judging convertibility. Large enough to
# be representative, small enough that validation stays interactive on
# datasets with millions of rows.
SAMPLE_SIZE = 200

# Fraction of sampled values that must cast successfully for a column to
# count as convertible. Below 1.0 so a handful of stray entries doesn't
# reject an otherwise clean column; high enough that genuinely mixed data
# is still rejected.
CONVERSION_THRESHOLD = 0.95

# Minimum observations before a dataset is considered to have enough
# history to forecast at all. Section 10's ≥24-period recommendation is
# the quality bar; this is the far lower floor below which forecasting is
# not meaningful.
MIN_OBSERVATIONS = 12

# Grains finer than monthly are aggregated up by the preprocessing pipeline
# before forecasting; the user chooses how (Sum/Mean/Last).
SUB_MONTHLY_FREQUENCIES = frozenset({"Daily", "Weekly"})

# Grains coarser than monthly cannot reach month level without
# disaggregating below the source grain, which the platform does not do.
COARSER_THAN_MONTHLY_FREQUENCIES = frozenset({"Quarterly", "Yearly"})


class ValidationEngine:
    """Produces the Metadata Validation Report and the final suitability verdict."""

    def validate(
        self, dataframe: pd.DataFrame, config: NormalizedMetadataConfig, dataset_name: str
    ) -> MetadataValidationResponse:
        """The full validation report plus the forecast-suitability verdict."""
        checks = [
            self._validate_date_column(dataframe, config),
            self._validate_target_column(dataframe, config),
            self._validate_business_keys(dataframe, config),
            self._validate_frequency(config),
        ]

        suitability = self._evaluate_suitability(dataframe, config, checks)
        ready_for_deployment = suitability.status is not SuitabilityStatus.NOT_SUITABLE

        summary = ForecastConfigurationSummary(
            dataset_name=dataset_name,
            rows=config.dataset_shape.rows,
            columns=config.dataset_shape.columns,
            forecast_mode=config.mode,
            forecast_frequency=config.forecast_frequency,
            unique_business_keys=config.unique_keys,
            aggregation_required=config.forecast_frequency in SUB_MONTHLY_FREQUENCIES,
        )

        return MetadataValidationResponse(
            mode=config.mode,
            normalized_config=config,
            checks=checks,
            forecast_suitability=suitability,
            ready_for_deployment=ready_for_deployment,
            configuration_summary=summary,
        )

    # ------------------------------------------------------------------
    # Individual validations
    # ------------------------------------------------------------------

    def _validate_date_column(
        self, dataframe: pd.DataFrame, config: NormalizedMetadataConfig
    ) -> ValidationCheckItem:
        """Judge whether the selected column can serve as the timeline."""
        column = config.date_column

        if column not in dataframe.columns:
            return self._check(
                "date-column", "Date Column", ValidationStatus.INVALID,
                f"'{column}' was not found in the dataset.",
            )

        series = dataframe[column]

        # Already a real datetime — nothing to convert.
        if pd.api.types.is_datetime64_any_dtype(series):
            return self._check(
                "date-column", "Date Column", ValidationStatus.VALID,
                f"'{column}' is stored as a datetime column and is ready for forecasting.",
            )

        sample = self._sample(series)
        if sample.empty:
            return self._check(
                "date-column", "Date Column", ValidationStatus.INVALID,
                f"'{column}' contains no values to validate.",
            )

        # Numeric columns are rejected rather than parsed: pandas would
        # happily read integers as nanosecond epochs and produce plausible
        # but meaningless timestamps, which is worse than a clear refusal.
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            return self._check(
                "date-column", "Date Column", ValidationStatus.INVALID,
                f"'{column}' holds numeric values and cannot be interpreted as a datetime field.",
            )

        success_ratio = self._parse_ratio(sample, self._to_datetime)
        if success_ratio >= CONVERSION_THRESHOLD:
            return self._check(
                "date-column", "Date Column", ValidationStatus.CONVERTIBLE,
                f"'{column}' is stored as text but can be safely converted into datetime "
                f"during preprocessing. Conversion will be performed automatically after deployment.",
            )

        return self._check(
            "date-column", "Date Column", ValidationStatus.INVALID,
            f"'{column}' cannot be interpreted as a datetime field.",
        )

    def _validate_target_column(
        self, dataframe: pd.DataFrame, config: NormalizedMetadataConfig
    ) -> ValidationCheckItem:
        """Judge whether the selected column can serve as the forecast target."""
        column = config.target_column

        if column not in dataframe.columns:
            return self._check(
                "target-column", "Target Column", ValidationStatus.INVALID,
                f"'{column}' was not found in the dataset.",
            )

        series = dataframe[column]
        sample = self._sample(series)

        if sample.empty:
            return self._check(
                "target-column", "Target Column", ValidationStatus.INVALID,
                f"'{column}' is empty and cannot be forecast.",
            )

        if pd.api.types.is_datetime64_any_dtype(series):
            return self._check(
                "target-column", "Target Column", ValidationStatus.INVALID,
                f"'{column}' holds datetime values and cannot be used as a numeric forecast target.",
            )

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            # A target that never varies cannot be meaningfully forecast —
            # the same flat-series signal forward-validation looks for.
            if sample.nunique() <= 1:
                return self._check(
                    "target-column", "Target Column", ValidationStatus.INVALID,
                    f"'{column}' is constant and does not contain enough variance to forecast.",
                )
            return self._check(
                "target-column", "Target Column", ValidationStatus.VALID,
                f"'{column}' is numeric and suitable for forecasting.",
            )

        success_ratio = self._parse_ratio(sample, self._to_numeric)
        if success_ratio >= CONVERSION_THRESHOLD:
            return self._check(
                "target-column", "Target Column", ValidationStatus.CONVERTIBLE,
                f"'{column}' is stored as text but can be converted into numeric values "
                f"during preprocessing.",
            )

        return self._check(
            "target-column", "Target Column", ValidationStatus.INVALID,
            f"'{column}' cannot be converted into the numeric values required for forecasting.",
        )

    def _validate_business_keys(
        self, dataframe: pd.DataFrame, config: NormalizedMetadataConfig
    ) -> ValidationCheckItem:
        """Report how many forecasting groups the key selection produces."""
        if not config.key_columns:
            return self._check(
                "business-keys", "Business Keys", ValidationStatus.VALID,
                "No key columns selected — the dataset will be forecast as a single series.",
            )

        missing = [column for column in config.key_columns if column not in dataframe.columns]
        if missing:
            return self._check(
                "business-keys", "Business Keys", ValidationStatus.INVALID,
                f"Key column(s) {', '.join(missing)} were not found in the dataset.",
            )

        if config.unique_keys == 0:
            return self._check(
                "business-keys", "Business Keys", ValidationStatus.INVALID,
                "The selected key column(s) produced no forecasting groups.",
            )

        return self._check(
            "business-keys", "Business Keys", ValidationStatus.VALID,
            f"{config.unique_keys:,} unique forecasting group(s) detected from {config.composite_key}.",
        )

    def _validate_frequency(self, config: NormalizedMetadataConfig) -> ValidationCheckItem:
        """Report the detected grain.

        Sub-monthly is CONVERTIBLE, not VALID: the pipeline aggregates it up
        to monthly, and the user should know that is coming. Coarser than
        monthly is rejected outright, since that would need disaggregation.
        """
        frequency = config.forecast_frequency

        if frequency in {"Unknown", "Irregular"}:
            return self._check(
                "frequency", "Frequency", ValidationStatus.CONVERTIBLE,
                f"A consistent frequency could not be detected from '{config.date_column}'. "
                f"The dataset can still be processed, but forecast confidence may be lower.",
                status_label="Irregular",
            )

        if frequency in COARSER_THAN_MONTHLY_FREQUENCIES:
            return self._check(
                "frequency", "Frequency", ValidationStatus.INVALID,
                f"{frequency} data is coarser than the monthly grain this platform forecasts at. "
                f"The dataset is not supported because the platform does not disaggregate data "
                f"below its source grain.",
                status_label=frequency,
            )

        if frequency in SUB_MONTHLY_FREQUENCIES:
            return self._check(
                "frequency", "Frequency", ValidationStatus.CONVERTIBLE,
                f"{frequency} frequency detected. This platform forecasts at month level, so "
                f"the target column will be aggregated to monthly during preprocessing using "
                f"the aggregation method selected below.",
                status_label=frequency,
            )

        return self._check(
            "frequency", "Frequency", ValidationStatus.VALID,
            f"{frequency} frequency successfully detected.",
            status_label=frequency,
        )

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------

    def _evaluate_suitability(
        self,
        dataframe: pd.DataFrame,
        config: NormalizedMetadataConfig,
        checks: list[ValidationCheckItem],
    ) -> ForecastSuitability:
        """One verdict from every check: INVALID blocks deployment, CONVERTIBLE
        only warns, since preprocessing handles it automatically."""
        blocking = [check.description for check in checks if check.status is ValidationStatus.INVALID]
        warnings = [check.description for check in checks if check.status is ValidationStatus.CONVERTIBLE]

        # History length is judged here rather than as its own row: it is a
        # property of the dataset as a whole, not of any single column.
        if len(dataframe) < MIN_OBSERVATIONS:
            blocking.append(
                f"Insufficient historical observations — the dataset holds {len(dataframe):,} row(s), "
                f"below the minimum of {MIN_OBSERVATIONS}."
            )

        if blocking:
            return ForecastSuitability(
                status=SuitabilityStatus.NOT_SUITABLE,
                summary="Dataset is not suitable for forecasting.",
                reasons=blocking,
            )

        if warnings:
            return ForecastSuitability(
                status=SuitabilityStatus.WARNINGS,
                summary="Dataset is suitable for forecasting, with conversions applied during preprocessing.",
                reasons=warnings,
            )

        return ForecastSuitability(
            status=SuitabilityStatus.READY,
            summary="Dataset is suitable for forecasting.",
            reasons=[],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample(self, series: pd.Series) -> pd.Series:
        """A bounded sample drawn evenly across the column, so a file with clean
        first rows and malformed later ones is still judged honestly."""
        non_null = series.dropna()
        if len(non_null) <= SAMPLE_SIZE:
            return non_null

        step = len(non_null) // SAMPLE_SIZE
        return non_null.iloc[::step][:SAMPLE_SIZE]

    def _parse_ratio(self, sample: pd.Series, parser) -> float:
        """Fraction of `sample` that `parser` converts successfully."""
        if sample.empty:
            return 0.0
        return float(parser(sample).notna().mean())

    @staticmethod
    def _to_datetime(sample: pd.Series) -> pd.Series:
        """Attempt datetime conversion, marking failures as NaT."""
        return pd.to_datetime(sample, errors="coerce")

    @staticmethod
    def _to_numeric(sample: pd.Series) -> pd.Series:
        """Attempt numeric conversion, marking failures as NaN."""
        return pd.to_numeric(sample, errors="coerce")

    @staticmethod
    def _check(
        check_id: str,
        title: str,
        status: ValidationStatus,
        description: str,
        status_label: str | None = None,
    ) -> ValidationCheckItem:
        """Build one report row."""
        return ValidationCheckItem(
            id=check_id,
            title=title,
            status=status,
            status_label=status_label,
            description=description,
        )
