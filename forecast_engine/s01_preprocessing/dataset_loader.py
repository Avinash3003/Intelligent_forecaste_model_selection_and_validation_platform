"""Turns a file into a DataFrame — the only place that knows file formats.

Carries no business logic: it does not know what a date or target is, which
keeps format concerns out of the forecasting stages.

Deliberately mirrors the backend's own loader rather than importing it — the
engine stays independently deployable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecast_engine.utils.exceptions import DatasetLoadError

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".xlsx", ".xls"})


class DatasetLoader:
    """Reads CSV/Excel datasets into DataFrames."""

    # Load and validate a dataset file into a DataFrame
    def load(self, file_path: str | Path) -> pd.DataFrame:
        path = Path(file_path)

        if not path.exists():
            raise DatasetLoadError(f"Dataset '{path}' could not be found.")

        if not path.is_file():
            raise DatasetLoadError(f"Dataset path '{path}' is not a file.")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise DatasetLoadError(
                f"Unsupported file format '{suffix}'. Supported formats are: {supported}."
            )

        # Format branching lives here and nowhere else in the engine.
        try:
            dataframe = pd.read_csv(path) if suffix == ".csv" else pd.read_excel(path)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean domain error
            raise DatasetLoadError(f"Failed to read '{path.name}': {exc}") from exc

        # Fail fast on an empty dataset rather than letting later stages
        # crash on absent columns with a far less obvious message.
        if dataframe.empty or dataframe.shape[1] == 0:
            raise DatasetLoadError(f"'{path.name}' does not contain any usable data.")

        return dataframe
