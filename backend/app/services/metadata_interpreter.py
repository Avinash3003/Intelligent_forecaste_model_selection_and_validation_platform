"""Metadata Interpreter — Step 2 of the metadata pipeline (Section 5.1.2 /
6.2 of the design doc).

Takes the raw metadata a user selected in the frontend (date/target/key/
feature columns) plus the uploaded dataset, and produces ONE normalized
configuration object. Every downstream stage — ValidationEngine today, and
the Databricks pipeline in a later phase — consumes this normalized shape
instead of re-deriving it, which is what keeps the whole platform dataset
agnostic: nothing in this class ever references a specific column name
like "Store" or "Sales".
"""

import pandas as pd

from app.schemas.metadata import DatasetShape, MetadataRequest, NormalizedMetadataConfig
from app.services.frequency_detector import FrequencyDetector

SINGLE_SERIES = "Single Series"
MULTI_SERIES = "Multi Series"


class MetadataInterpreter:
    """Builds a NormalizedMetadataConfig from user metadata + a DataFrame."""

    def __init__(self, frequency_detector: FrequencyDetector | None = None) -> None:
        self._frequency_detector = frequency_detector or FrequencyDetector()

    def interpret(self, dataframe: pd.DataFrame, request: MetadataRequest) -> NormalizedMetadataConfig:
        """Interpret `request` against `dataframe` and normalize it.

        Args:
            dataframe: The uploaded dataset, already loaded by DatasetLoader.
            request: The raw column-role selections from the frontend.

        Returns:
            A NormalizedMetadataConfig capturing forecast mode, frequency,
            unique key count and dataset shape. Called by the
            /metadata/validate route, before ValidationEngine runs.
        """
        # The presence of key columns is the single deciding factor between
        # single-series and multi-series forecasting (Section 6.1) — no
        # other signal is needed to make this call.
        mode = MULTI_SERIES if request.key_columns else SINGLE_SERIES
        composite_key = " + ".join(request.key_columns) if request.key_columns else None

        forecast_frequency = self._detect_frequency(dataframe, request.date_column)
        unique_keys = self._count_unique_keys(dataframe, request.key_columns)

        return NormalizedMetadataConfig(
            date_column=request.date_column,
            target_column=request.target_column,
            key_columns=request.key_columns,
            feature_columns=request.feature_columns,
            mode=mode,
            composite_key=composite_key,
            forecast_frequency=forecast_frequency,
            unique_keys=unique_keys,
            dataset_shape=DatasetShape(rows=dataframe.shape[0], columns=dataframe.shape[1]),
        )

    def _detect_frequency(self, dataframe: pd.DataFrame, date_column: str) -> str:
        """Return the detected frequency for `date_column`, or "Unknown" if
        the column doesn't exist in the dataset.

        The interpreter must never raise on a bad column reference —
        ValidationEngine is responsible for reporting that as a proper
        validation failure with a user-facing message.
        """
        if date_column not in dataframe.columns:
            return "Unknown"
        return self._frequency_detector.detect(dataframe[date_column])

    def _count_unique_keys(self, dataframe: pd.DataFrame, key_columns: list[str]) -> int:
        """Count distinct business keys formed by `key_columns`.

        A single-series dataset (no key columns selected) is treated as one
        implicit key. Missing key columns are ignored here — surfaced as a
        validation error instead — so this always returns a number instead
        of raising.
        """
        if not key_columns:
            return 1

        existing_columns = [column for column in key_columns if column in dataframe.columns]
        if not existing_columns:
            return 0

        # Distinct rows across the selected key columns == distinct
        # business keys (e.g. every unique Store + SKU combination).
        return int(dataframe[existing_columns].drop_duplicates().shape[0])
