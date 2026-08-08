"""Forecast Configuration — the engine's single input contract.

This is the object the backend's Metadata Interpreter produces and the
Forecast Engine consumes. It is the *only* place column identities live:
every stage downstream asks this object which column is the date, the
target, a key or a feature, and never assumes a name like "Store",
"Sales" or "SKU". That indirection is what makes the same engine run
against completely different datasets with zero code changes — the
portability bar set in Section 14.1.

The object is immutable: it travels through every pipeline stage, and a
stage silently rewriting it would make a run impossible to audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from forecast_engine.utils.exceptions import ConfigurationError


class ForecastMode(str, Enum):
    """Whether the dataset is forecast as one series or many.

    The distinction drives grouping, and later ranking and drift
    validation, which all operate per business key.
    """

    SINGLE_SERIES = "Single Series"
    MULTI_SERIES = "Multi Series"


class AggregationMethod(str, Enum):
    """How a sub-monthly target is rolled up to the monthly grain.

    Forecasting runs at month level, so Daily/Weekly datasets are
    aggregated during preprocessing. Which method is correct depends on
    what the target measures — SUM for flow quantities, MEAN or LAST for
    stock-type levels — so the user selects it rather than the platform
    guessing.
    """

    SUM = "sum"
    MEAN = "mean"
    LAST = "last"


@dataclass(frozen=True)
class ForecastConfiguration:
    """Normalized metadata describing how to forecast a dataset.

    Attributes:
        date_column: Column holding each observation's timestamp.
        target_column: Numeric column to be forecast.
        key_columns: Columns that together identify one forecasting
            entity. Empty means the dataset is a single series.
        feature_columns: Optional exogenous regressors to carry into
            training alongside the target.
    """

    date_column: str
    target_column: str
    key_columns: tuple[str, ...] = ()
    feature_columns: tuple[str, ...] = ()

    # Applied by preprocessing only when the detected grain is finer than
    # monthly; ignored for data already at month level.
    aggregation_method: AggregationMethod = AggregationMethod.SUM

    # Forecast mode implied by the metadata (key columns present -> multi-series)
    @property
    def mode(self) -> ForecastMode:
        return ForecastMode.MULTI_SERIES if self.key_columns else ForecastMode.SINGLE_SERIES

    # Human-readable business key description, for display/logging only
    @property
    def composite_key(self) -> str | None:
        return " + ".join(self.key_columns) if self.key_columns else None

    # Every column the dataset must actually contain for this run
    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.date_column, self.target_column, *self.key_columns, *self.feature_columns)

    # Columns making up a built time series: timeline, target, regressors
    @property
    def series_columns(self) -> tuple[str, ...]:
        return (self.date_column, self.target_column, *self.feature_columns)

    # Build a configuration from the backend's normalized metadata
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ForecastConfiguration":
        # Derived fields present in the backend response — mode, frequency,
        # unique key counts — are ignored; the engine recomputes them from
        # the real dataset rather than trusting a value computed elsewhere.
        try:
            date_column = payload["date_column"]
            target_column = payload["target_column"]
        except KeyError as exc:
            raise ConfigurationError(f"Forecast configuration is missing required field {exc}.") from exc

        try:
            aggregation_method = AggregationMethod(payload.get("aggregation_method") or "sum")
        except ValueError as exc:
            supported = ", ".join(method.value for method in AggregationMethod)
            raise ConfigurationError(
                f"Unsupported aggregation method '{payload.get('aggregation_method')}'. "
                f"Supported methods are: {supported}."
            ) from exc

        configuration = cls(
            date_column=date_column,
            target_column=target_column,
            key_columns=tuple(payload.get("key_columns") or ()),
            feature_columns=tuple(payload.get("feature_columns") or ()),
            aggregation_method=aggregation_method,
        )
        configuration.validate()
        return configuration

    # Check the configuration is internally consistent
    def validate(self) -> None:
        # Verifies every role is populated and no column is assigned two
        # roles — a column cannot be both the target and a key, for example.
        if not self.date_column or not self.target_column:
            raise ConfigurationError("Both a date column and a target column must be specified.")

        if self.date_column == self.target_column:
            raise ConfigurationError(
                f"Column '{self.date_column}' cannot be both the date and the target column."
            )

        # Count every role assignment; any column appearing more than once
        # holds conflicting roles.
        seen: dict[str, int] = {}
        for column in self.required_columns:
            seen[column] = seen.get(column, 0) + 1

        conflicts = sorted(column for column, count in seen.items() if count > 1)
        if conflicts:
            raise ConfigurationError(
                f"Column(s) {', '.join(conflicts)} are assigned to more than one metadata role."
            )

    # Check every declared column exists in the loaded dataset
    def validate_against_columns(self, available_columns: list[str]) -> None:
        # Lists every missing column at once, so an operator fixes the
        # mapping in a single pass rather than rerunning to discover each.
        available = set(available_columns)
        missing = [column for column in self.required_columns if column not in available]
        if missing:
            raise ConfigurationError(
                f"Column(s) {', '.join(missing)} declared in the forecast configuration "
                f"are not present in the dataset."
            )

    # Serialize for logging and run artifacts, including the derived mode
    def to_dict(self) -> dict[str, Any]:
        return {
            "date_column": self.date_column,
            "target_column": self.target_column,
            "key_columns": list(self.key_columns),
            "feature_columns": list(self.feature_columns),
            "mode": self.mode.value,
            "composite_key": self.composite_key,
            "aggregation_method": self.aggregation_method.value,
        }
