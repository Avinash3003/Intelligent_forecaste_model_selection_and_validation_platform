"""Schema for the uploaded-dataset preview shown on the Results page."""

from __future__ import annotations

from pydantic import BaseModel


class DatasetPreview(BaseModel):
    available: bool = False
    dataset_name: str | None = None
    # Where the bytes came from — surfaced so a user can tell an Azure-backed
    # run from a locally-staged one without checking configuration.
    source: str | None = None
    columns: list[str] = []
    rows: list[list[str]] = []
    preview_row_count: int = 0
    # True when the file is larger than the previewed slice.
    truncated: bool = False
    status: str | None = None
