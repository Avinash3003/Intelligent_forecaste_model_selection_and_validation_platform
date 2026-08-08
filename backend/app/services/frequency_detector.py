"""Forecast Frequency Detection — Step 4 of the metadata pipeline.

Infers the sampling grain (Daily / Weekly / Monthly / Quarterly / Yearly) of
a date column purely from the spacing between its own timestamps. This is
detection only — no resampling or normalization happens here. That belongs
to a later ingestion-stage pipeline component (Section 6.2, "Grain
normalization to monthly"), which is out of scope for this phase.
"""

import pandas as pd

# Median gap between consecutive observed dates, in days, mapped to a
# human-readable grain. Ranges are intentionally generous so a handful of
# missing periods or weekend-skipping data doesn't misclassify the
# frequency.
_FREQUENCY_BANDS: list[tuple[float, float, str]] = [
    (0, 1, "Daily"),
    (2, 9, "Weekly"),
    (10, 45, "Monthly"),
    (46, 135, "Quarterly"),
    (136, 400, "Yearly"),
]


class FrequencyDetector:
    """Detects the forecast frequency implied by a date column's values."""

    def detect(self, date_series: pd.Series) -> str:
        """Infer the sampling grain of `date_series`.

        Args:
            date_series: Raw values from the user-selected date column
                (may contain unparseable or duplicate values).

        Returns:
            One of "Daily", "Weekly", "Monthly", "Quarterly", "Yearly", or
            "Irregular" if no consistent spacing can be determined. Called
            by MetadataInterpreter while building the normalized config.
        """
        parsed = pd.to_datetime(date_series, errors="coerce").dropna()
        unique_sorted = parsed.drop_duplicates().sort_values()

        if len(unique_sorted) < 2:
            return "Irregular"

        # The median gap is robust to a handful of missing/irregular
        # periods, which real-world datasets almost always have — a mean
        # would be skewed by a single large gap (e.g. a data outage).
        gaps_in_days = unique_sorted.diff().dropna().dt.days
        median_gap = float(gaps_in_days.median())

        for lower, upper, label in _FREQUENCY_BANDS:
            if lower <= median_gap <= upper:
                return label

        return "Irregular"
