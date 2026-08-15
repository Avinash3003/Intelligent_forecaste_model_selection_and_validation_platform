"""Turns a staged file into a DataFrame, and nothing else.

Knows nothing about date/target/key columns — that is MetadataInterpreter's
job one layer up. Keeping the boundary clean means changing where files are
stored touches only this and UploadService.
"""

from pathlib import Path

import pandas as pd

from app.utils.exceptions import DatasetLoadError

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class DatasetLoader:
    """Reads an uploaded CSV/Excel file into a pandas DataFrame."""

    def load(self, file_path: Path) -> pd.DataFrame:
        """Load a staged file into a non-empty DataFrame.

        Raises DatasetLoadError if it is missing, an unsupported type,
        unparseable, or empty.
        """
        if not file_path.exists():
            raise DatasetLoadError(f"Uploaded file '{file_path.name}' could not be found on the server.")

        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise DatasetLoadError(
                f"Unsupported file format '{suffix}'. Supported formats are CSV, XLSX and XLS."
            )

        # Route to the correct pandas reader based on extension — this is
        # the only place file-format branching happens in the pipeline.
        try:
            if suffix == ".csv":
                dataframe = pd.read_csv(file_path)
            else:
                dataframe = pd.read_excel(file_path)
        except Exception as exc:  # noqa: BLE001 - re-raised as a clean domain error below
            raise DatasetLoadError(f"Failed to read '{file_path.name}': {exc}") from exc

        # An empty file (no rows or no columns) can't be interpreted or
        # validated, so fail fast here with a clear message instead of
        # letting later stages crash on missing columns.
        if dataframe.empty or dataframe.shape[1] == 0:
            raise DatasetLoadError(f"'{file_path.name}' does not contain any usable data.")

        return dataframe
