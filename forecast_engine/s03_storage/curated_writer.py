"""Curated dataset persistence.

The curated dataset is the single input every later forecasting stage
reads, so where it lives is an infrastructure decision that must not leak
into business logic. This module defines a narrow backend interface plus a
local-disk implementation; swapping in ADLS Gen2 or Blob Storage later
means adding one class here and selecting it in configuration — no stage
that produces or consumes curated data changes.

The raw uploaded dataset is never written to. Curated output always lands
in a separate location, keyed by run id, so a run's input remains
reproducible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from forecast_engine.config.pipeline_config import CuratedStorageConfig
from forecast_engine.utils.exceptions import CuratedDatasetError

SUPPORTED_FORMATS: frozenset[str] = frozenset({"csv", "parquet"})


class CuratedDatasetBackend(ABC):
    """Destination for curated datasets.

    Implementations are responsible only for *where* and *how* bytes are
    persisted. They never inspect or alter the data, which is what keeps
    storage swappable.
    """

    # Persist a dataframe and return a URI identifying what was written
    @abstractmethod
    def write(self, dataframe: pd.DataFrame, relative_path: str, file_format: str) -> str:
        ...


class LocalCuratedBackend(CuratedDatasetBackend):
    """Writes curated datasets to the local filesystem.

    Used for local development and for the pipeline's own tests. The
    directory layout it creates (`<root>/<run_id>/<name>`) is intentionally
    the same shape a blob container would use, so moving to cloud storage is
    a backend swap rather than a rethink.
    """

    # Store the local root directory
    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)

    # Write a dataframe to local disk as CSV or parquet
    def write(self, dataframe: pd.DataFrame, relative_path: str, file_format: str) -> str:
        destination = self._root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            if file_format == "parquet":
                dataframe.to_parquet(destination, index=False)
            else:
                dataframe.to_csv(destination, index=False)
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
