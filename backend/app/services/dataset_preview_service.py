"""Dataset preview for the Results page.

Shows the *curated* dataset — the cleaned, deduplicated, monthly-aggregated
data every model actually trained on — rather than the raw upload. This is a
deliberate choice, not just a convenience:

  * It is what a "does this look right" check should be judging. The raw
    upload can carry duplicates, bad rows and a finer grain than the platform
    forecasts at; the curated file is post-preprocessing, so it is the same
    data the decision on this page was actually made from.
  * It is far smaller. A daily upload across many keys aggregates down to one
    row per key per month, so pagination cost stays low without needing the
    raw file's byte-range machinery.

Curated storage is local-disk only today (Section 6.2/6.14) — there is no
cloud backend yet — so this reads a plain file path, resolved from the run's
own summary, and never touches Azure.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

from app.orchestration.executor import PipelineExecutor, get_pipeline_executor
from app.schemas.dataset_preview import DatasetPreview

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

# Parsed curated files are small (see module docstring) but re-parsing on
# every page click is still wasted work, so the last few runs a user looked
# at stay cached. Bounded so browsing many runs in one session cannot grow
# this without limit.
_CACHE_CAPACITY = 5


class DatasetPreviewService:
    """Reads a run's curated dataset, one page of rows at a time."""

    def __init__(self, executor: PipelineExecutor | None = None) -> None:
        self._executor = executor or get_pipeline_executor()
        self._cache: "OrderedDict[str, tuple[list[str], list[list[str]]]]" = OrderedDict()

    def get_preview(self, run_id: str, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE) -> DatasetPreview:
        page = max(page, 1)
        page_size = min(max(page_size, 1), MAX_PAGE_SIZE)

        parsed = self._load(run_id)
        if parsed is None:
            return DatasetPreview(
                available=False,
                status="No curated dataset was recorded for this run — curated storage may be disabled.",
            )
        columns, all_rows = parsed

        total_rows = len(all_rows)
        total_pages = max(1, -(-total_rows // page_size))  # ceiling division
        start = (page - 1) * page_size
        page_rows = all_rows[start : start + page_size]

        return DatasetPreview(
            available=True,
            columns=columns,
            rows=page_rows,
            page=page,
            page_size=page_size,
            total_rows=total_rows,
            total_pages=total_pages,
        )

    # ------------------------------------------------------------------
    # Loading (cached per run)
    # ------------------------------------------------------------------

    def _load(self, run_id: str) -> tuple[list[str], list[list[str]]] | None:
        if run_id in self._cache:
            self._cache.move_to_end(run_id)
            return self._cache[run_id]

        path = self._curated_path(run_id)
        if path is None or not path.is_file():
            return None

        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            try:
                columns = next(reader)
            except StopIteration:
                return [], []
            rows = list(reader)

        self._cache[run_id] = (columns, rows)
        if len(self._cache) > _CACHE_CAPACITY:
            self._cache.popitem(last=False)
        return columns, rows

    def _curated_path(self, run_id: str) -> Path | None:
        try:
            result = self._executor.get_result(run_id)
        except Exception:  # noqa: BLE001 - an unreadable run simply has no preview
            return None
        uri = (result.run_metadata or {}).get("curated_dataset_uri")
        return Path(uri) if uri else None


_service: DatasetPreviewService | None = None


def get_dataset_preview_service() -> DatasetPreviewService:
    global _service
    if _service is None:
        _service = DatasetPreviewService()
    return _service
