"""Turns the user's column choices into one normalized config.

Everything downstream reads this shape instead of re-deriving it, which is
what keeps the platform dataset-agnostic — nothing here ever names a
specific column like "Store" or "Sales".
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
        """Normalize the frontend's column choices into forecast mode,
        frequency, key count and dataset shape."""
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
        """Detected frequency, or "Unknown" for a missing column.

        Never raises on a bad column — validation reports that properly.
        """
        if date_column not in dataframe.columns:
            return "Unknown"
        return self._frequency_detector.detect(dataframe[date_column])

    def _count_unique_keys(self, dataframe: pd.DataFrame, key_columns: list[str]) -> int:
        """Count distinct business keys; no key columns means one implicit key.

        Missing columns are ignored here and reported by validation instead.
        """
        if not key_columns:
            return 1

        existing_columns = [column for column in key_columns if column in dataframe.columns]
        if not existing_columns:
            return 0

        # Distinct rows across the selected key columns == distinct
        # business keys (e.g. every unique Store + SKU combination).
        return int(dataframe[existing_columns].drop_duplicates().shape[0])
