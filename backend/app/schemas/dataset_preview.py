"""Schema for the curated-dataset preview shown on the Results page."""

from __future__ import annotations

from pydantic import BaseModel


class DatasetPreview(BaseModel):
    available: bool = False
    columns: list[str] = []
    rows: list[list[str]] = []
    # 1-indexed, matching how the frontend labels pages.
    page: int = 1
    page_size: int = 50
    total_rows: int = 0
    total_pages: int = 1
    status: str | None = None
