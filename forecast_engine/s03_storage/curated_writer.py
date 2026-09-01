"""Where the curated dataset is written.

A narrow backend interface plus a local-disk implementation, so swapping in
blob storage means adding one class here — no stage that produces or
consumes curated data changes.

The raw upload is never written to: curated output lands in its own
location keyed by run id, so a run's input stays reproducible.
"""

from __future__ import annotations

import io

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from forecast_engine.config.pipeline_config import CuratedStorageConfig
from forecast_engine.core import storage
from forecast_engine.utils.exceptions import CuratedDatasetError

SUPPORTED_FORMATS: frozenset[str] = frozenset({"csv", "parquet"})


class CuratedDatasetBackend(ABC):
    """Where and how bytes are persisted. Implementations never inspect or
    alter the data, which is what keeps storage swappable."""

    # Persist a dataframe and return a URI identifying what was written
    @abstractmethod
    def write(self, dataframe: pd.DataFrame, relative_path: str, file_format: str) -> str:
        ...


class LocalCuratedBackend(CuratedDatasetBackend):
    """Writes to local disk, using the same <root>/<run_id>/<name> layout a
    blob container would — so moving to the cloud is a backend swap."""

    # Store the local root directory
    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)

    # Write a dataframe to local disk as CSV or parquet
    def write(self, dataframe: pd.DataFrame, relative_path: str, file_format: str) -> str:
        destination = self._root / relative_path
        storage.ensure_parent(destination)

        try:
            # Serialised to a buffer and written once through the adapter.
            # pandas writes to a path, which a DCS container cannot open on
            # a UC Volume; the bytes are identical either way.
            buffer = io.BytesIO()
            if file_format == "parquet":
                dataframe.to_parquet(buffer, index=False)
            else:
                buffer.write(dataframe.to_csv(index=False).encode("utf-8"))
            storage.write_bytes(destination, buffer.getvalue())
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            raise CuratedDatasetError(f"Failed to write curated dataset to '{destination}': {exc}") from exc

        return str(destination.resolve())


class CuratedDatasetWriter:
    """Names and persists the curated dataset for a run.

    Owns the naming convention so every backend produces the same layout,
    and so a curated file can always be traced back to the run that made it.
    """

    # Store storage config and backend
    def __init__(
        self,
        config: CuratedStorageConfig | None = None,
        backend: CuratedDatasetBackend | None = None,
    ) -> None:
        self._config = config or CuratedStorageConfig()
        self._backend = backend or LocalCuratedBackend(self._config.root_dir)

    # Persist the curated dataset for one run
    def write(self, dataframe: pd.DataFrame, run_id: str, source_name: str) -> str:
        file_format = self._config.file_format.lower()
        if file_format not in SUPPORTED_FORMATS:
            raise CuratedDatasetError(
                f"Unsupported curated output format '{file_format}'. "
                f"Supported formats are: {', '.join(sorted(SUPPORTED_FORMATS))}."
            )

        filename = f"curated_{Path(source_name).stem}.{file_format}"
        return self._backend.write(dataframe, f"{run_id}/{filename}", file_format)


# Read a curated dataset back from its writer-returned URI. Needed only by a
# Databricks task resuming from a checkpoint — a single process never needs
# this, since it still holds the DataFrame it just wrote. `date_column` is
# only used for the csv path: parquet keeps its own dtypes, but csv.to_csv()
# wrote the date column as plain ISO text, and pd.read_csv leaves it a str
# unless told which column to parse back to a timestamp.
def read_curated_dataset(uri: str, date_column: str) -> pd.DataFrame:
    file_format = Path(uri).suffix.lstrip(".").lower()
    if file_format not in SUPPORTED_FORMATS:
        raise CuratedDatasetError(f"Cannot read curated dataset with unrecognised format '{file_format}' ({uri}).")
    with storage.open_binary(uri) as handle:
        if file_format == "parquet":
            return pd.read_parquet(handle)
        return pd.read_csv(handle, parse_dates=[date_column])
