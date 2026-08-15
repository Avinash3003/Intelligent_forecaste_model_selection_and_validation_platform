"""The assessment stage's output shape.

Pure data, no logic. Every field is a plain JSON type (no numpy scalars, no
DataFrames) so to_dict() is always safe to hand to json.dumps.

Kept separate from the assessor so the contract the UI depends on can be
read in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SuitabilityStatus(str, Enum):
    """Whether a dataset can be forecast, using the same three states the
    backend's validation engine reports."""

    READY = "Ready"
    WARNINGS = "Warnings"
    NOT_SUITABLE = "Not Suitable"


@dataclass
class ColumnQuality:
    """Per-column quality facts, derived without modifying anything."""

    name: str
    dtype: str
    missing_values: int
    null_percentage: float
    distinct_values: int

    # Serialize column quality facts to a dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "missing_values": self.missing_values,
            "null_percentage": round(self.null_percentage, 2),
            "distinct_values": self.distinct_values,
        }


@dataclass
class QualityReport:
    """Everything measured about the raw dataset, before any cleaning.

    duplicate_timestamps counts rows sharing a key and a timestamp — a
    time-series defect a plain duplicate check misses, since the target
    values may differ. invalid_* are values present but unconvertible.
    missing_timestamps is None when the grain is irregular or unscanned.
    """

    total_rows: int = 0
    total_columns: int = 0
    duplicate_rows: int = 0
    duplicate_timestamps: int = 0
    columns: list[ColumnQuality] = field(default_factory=list)

    invalid_date_values: int = 0
    invalid_target_values: int = 0
    constant_target: bool = False

    distinct_business_keys: int = 0
    frequency: str | None = None
    date_range_start: str | None = None
    date_range_end: str | None = None
    total_observations: int = 0
    missing_timestamps: int | None = None

    suitability: SuitabilityStatus = SuitabilityStatus.READY
    suitability_reasons: list[str] = field(default_factory=list)

    # Whether the dataset may proceed to preprocessing
    @property
    def is_forecastable(self) -> bool:
        # Warnings do not block: they describe defects that preprocessing
        # repairs (duplicate rows, a handful of unconvertible values).
        return self.suitability is not SuitabilityStatus.NOT_SUITABLE

    # Serialize the full report for logging or frontend display
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_timestamps": self.duplicate_timestamps,
            "columns": [column.to_dict() for column in self.columns],
            "invalid_date_values": self.invalid_date_values,
            "invalid_target_values": self.invalid_target_values,
            "constant_target": self.constant_target,
            "distinct_business_keys": self.distinct_business_keys,
            "frequency": self.frequency,
            "date_range_start": self.date_range_start,
            "date_range_end": self.date_range_end,
            "total_observations": self.total_observations,
            "missing_timestamps": self.missing_timestamps,
            "suitability": self.suitability.value,
            "suitability_reasons": list(self.suitability_reasons),
        }


@dataclass
class PreprocessingSummary:
    """What preprocessing did, as actions taken rather than data described.

    Together with QualityReport this reconciles the raw row count with the
    curated one, line by line.
    """

    rows_read: int = 0
    rows_removed: int = 0
    duplicate_rows_removed: int = 0
    invalid_date_rows_removed: int = 0
    missing_target_rows_removed: int = 0

    date_converted: bool = False
    target_converted: bool = False

    # Monthly roll-up applied to sub-monthly data. Internal to the
    # preprocessing pipeline — recorded for traceability, not surfaced to
    # the frontend.
    aggregation_applied: bool = False
    aggregation_method: str | None = None
    rows_after_aggregation: int | None = None
    aggregation_detail: dict[str, Any] = field(default_factory=dict)

    detected_frequency: str | None = None
    forecast_mode: str | None = None
    forecast_ready: bool = False
    status: str = "Completed"

    curated_rows: int = 0
    curated_dataset_uri: str | None = None

    # Serialize preprocessing summary to a dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_read": self.rows_read,
            "rows_removed": self.rows_removed,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "invalid_date_rows_removed": self.invalid_date_rows_removed,
            "missing_target_rows_removed": self.missing_target_rows_removed,
            "date_converted": self.date_converted,
            "target_converted": self.target_converted,
            "aggregation_applied": self.aggregation_applied,
            "aggregation_method": self.aggregation_method,
            "rows_after_aggregation": self.rows_after_aggregation,
            "aggregation_detail": self.aggregation_detail,
            "detected_frequency": self.detected_frequency,
            "forecast_mode": self.forecast_mode,
            "forecast_ready": self.forecast_ready,
            "status": self.status,
            "curated_rows": self.curated_rows,
            "curated_dataset_uri": self.curated_dataset_uri,
        }
