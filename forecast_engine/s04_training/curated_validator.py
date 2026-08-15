"""A last guard before training that the curated dataset is what
preprocessing promised.

Checks for a non-empty frame, a real datetime column, a numeric target, one
observation per key per period, and enough history to fit anything.

Internal only — it produces no report and surfaces nothing to the frontend.
It exists to fail loudly if a bad curated dataset ever reaches training,
which would otherwise appear as a confusing library error inside a model.
"""

from __future__ import annotations

import pandas as pd

from forecast_engine.config.pipeline_config import QualityConfig
from forecast_engine.core.forecast_configuration import ForecastConfiguration
from forecast_engine.utils.exceptions import TrainingError


class CuratedDatasetValidator:
    """Verifies a curated dataset is fit to train on."""

    # Store quality thresholds used by validation checks
    def __init__(self, config: QualityConfig | None = None) -> None:
        self._config = config or QualityConfig()

    # Raise TrainingError if the curated dataset is not fit to train on
    def validate(self, dataframe: pd.DataFrame | None, configuration: ForecastConfiguration) -> None:
        if dataframe is None:
            raise TrainingError("No curated dataset is available — preprocessing must run before training.")

        if dataframe.empty:
            raise TrainingError("The curated dataset is empty; there is nothing to train on.")

        configuration.validate_against_columns(list(dataframe.columns))

        # Preprocessing standardizes both types; anything else means a
        # curated dataset was produced or loaded outside that guarantee.
        date_column = configuration.date_column
        if not pd.api.types.is_datetime64_any_dtype(dataframe[date_column]):
            raise TrainingError(
                f"Curated dataset column '{date_column}' is not a datetime column "
                f"(found {dataframe[date_column].dtype})."
            )

        target_column = configuration.target_column
        if not pd.api.types.is_numeric_dtype(dataframe[target_column]):
            raise TrainingError(
                f"Curated dataset column '{target_column}' is not numeric "
                f"(found {dataframe[target_column].dtype})."
            )

        # One observation per key per period is a structural requirement of
        # a time series; duplicates here would silently corrupt every fit.
        subset = [*configuration.key_columns, date_column]
        duplicate_timestamps = int(dataframe.duplicated(subset=subset).sum())
        if duplicate_timestamps:
            raise TrainingError(
                f"Curated dataset contains {duplicate_timestamps:,} duplicate timestamp(s) "
                f"within a forecasting group."
            )

        if len(dataframe) < self._config.min_observations:
            raise TrainingError(
                f"Curated dataset holds {len(dataframe):,} row(s), below the minimum of "
                f"{self._config.min_observations} required for training."
            )
