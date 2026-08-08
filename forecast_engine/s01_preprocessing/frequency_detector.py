"""Frequency Detector — infers a dataset's sampling grain.

Reads only the user-selected date column and reports which grain its
timestamps imply. This is detection, never conversion: the engine records
the frequency and preserves the data at its original grain. Normalizing to
a common grain is a separate ingestion concern (Section 6.2) and is
deliberately out of scope here, so that no stage silently changes the
values a model will later be judged on.

Frequency bands come from configuration, so a deployment can retune what
counts as "Weekly" without editing this logic.
"""

from __future__ import annotations

import pandas as pd

from forecast_engine.config.pipeline_config import FREQUENCY_BANDS, IRREGULAR_FREQUENCY


class FrequencyDetector:
    """Detects the forecast frequency implied by a column of timestamps."""

    # Store the configured frequency bands
    def __init__(self, bands: tuple[tuple[float, float, str], ...] = FREQUENCY_BANDS) -> None:
        self._bands = bands

    # Infer the sampling grain of a date column
    def detect(self, date_series: pd.Series) -> str:
        parsed = pd.to_datetime(date_series, errors="coerce").dropna()

        # Duplicates are expected in multi-series data (many keys share one
        # date), so distinct timestamps define the timeline, not row count.
        distinct_sorted = parsed.drop_duplicates().sort_values()

        # A single timestamp has no spacing to measure.
        if len(distinct_sorted) < 2:
            return IRREGULAR_FREQUENCY

        # The median gap is robust to occasional missing periods; a mean
        # would be dragged off by one large gap (e.g. a data outage).
        gaps_in_days = distinct_sorted.diff().dropna().dt.days
        median_gap = float(gaps_in_days.median())

        for lower, upper, label in self._bands:
            if lower <= median_gap <= upper:
                return label

        return IRREGULAR_FREQUENCY
