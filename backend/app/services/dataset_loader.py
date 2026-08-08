"""Dataset Loader — Step 1 of the metadata pipeline (Section 6.2 of the
solution design doc, "Data Ingestion & Validation").

Its only responsibility is turning a file on disk into a pandas DataFrame.
It knows nothing about forecasting metadata (date/target/key columns) —
that interpretation happens one layer up, in MetadataInterpreter. Keeping
this boundary clean means swapping local disk storage for ADLS / Blob
Storage in a later phase only touches UploadService + this loader, never
the interpretation/validation logic built on top of it.
"""

from pathlib import Path

import pandas as pd

from app.utils.exceptions import DatasetLoadError

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class DatasetLoader:
    """Reads an uploaded CSV/Excel file into a pandas DataFrame."""

    def load(self, file_path: Path) -> pd.DataFrame:
        """Load `file_path` into a DataFrame.

        Args:
            file_path: Absolute path to a file previously staged by
                UploadService.

        Returns:
            A non-empty pandas DataFrame.

        Raises:
            DatasetLoadError: if the file is missing, has an unsupported
                extension, fails to parse, or contains no usable data.
                Called from the /metadata/validate route, before any
                column-level validation runs.
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
