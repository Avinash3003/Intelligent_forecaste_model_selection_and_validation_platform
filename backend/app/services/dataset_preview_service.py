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

The curated file lives wherever the run that produced it wrote it: on the
local filesystem for local execution, and in the curated Unity Catalog
Volume for cloud execution. Both are plain POSIX paths in the run's own
summary; only *this* process's ability to open them differs, since a UC
Volume path means nothing on the API host. So a path that is not readable
locally is fetched through the same workspace client the runner already
uses for the run's other files.
"""

from __future__ import annotations

import csv
import io
import math
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from app.orchestration.executor import PipelineExecutor, get_pipeline_executor
from app.schemas.dataset_preview import DatasetPreview
from app.services.derived_feature_registry import (
    SUPPORTED_CALENDAR_FEATURES,
    SUPPORTED_DERIVED_FEATURE_IDS,
    SUPPORTED_LAGS,
    SUPPORTED_ROLLING_WINDOWS,
)

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
        columns, all_rows = self._with_derived_feature_columns(run_id, *parsed)

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

        uri = self._curated_uri(run_id)
        if not uri:
            return None

        text = self._read(uri)
        if text is None:
            return None

        reader = csv.reader(io.StringIO(text))
        try:
            columns = next(reader)
        except StopIteration:
            return [], []
        rows = list(reader)

        self._cache[run_id] = (columns, rows)
        if len(self._cache) > _CACHE_CAPACITY:
            self._cache.popitem(last=False)
        return columns, rows

    def _read(self, uri: str) -> str | None:
        """The curated file's contents, from wherever the run wrote it.

        Tries the local filesystem first, which is both the local-execution
        case and the cheaper one. A UC Volume path simply is not a local
        file, so that check fails and the runner's own workspace client
        fetches it instead.
        """
        path = Path(uri)
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
        return self._read_remote(uri)

    def _read_remote(self, uri: str) -> str | None:
        # Only a Databricks-backed runner can reach a UC Volume; a local
        # runner has no workspace client, and there is nothing to fall back
        # to, so the preview reports itself unavailable rather than raising.
        runner = getattr(self._executor, "_runner", None)
        read_volume_file = getattr(runner, "read_volume_text", None)
        if read_volume_file is None:
            return None
        try:
            return read_volume_file(uri)
        except Exception:  # noqa: BLE001 - an unreadable file simply has no preview
            return None

    def get_full_series(
        self, run_id: str, date_column: str, target_column: str, key_values: dict[str, object] | None = None
    ) -> list[tuple[str, float]] | None:
        """Every (date, target) observation for one business key, in this
        run's curated dataset — the complete history, never a bounded tail.

        Reuses the exact same cached curated-file parse `get_preview()`
        already uses (Priority A: the Results chart's "actual" line reads
        this instead of `PipelineContext.summary()`'s bounded
        `recent_history`, which exists only as a lightweight fallback).
        Returns None (not an empty list) when the curated file itself is
        unavailable, so a caller can distinguish "no curated data" from
        "no observations for this key".
        """
        parsed = self._load(run_id)
        if parsed is None:
            return None
        columns, all_rows = parsed
        if date_column not in columns or target_column not in columns:
            return None

        date_idx = columns.index(date_column)
        target_idx = columns.index(target_column)
        key_indices = {
            name: columns.index(name) for name in (key_values or {}) if name in columns
        }

        points: list[tuple[str, float]] = []
        for row in all_rows:
            if not self._row_matches_key(row, key_indices, key_values or {}):
                continue
            try:
                value = float(row[target_idx])
            except (TypeError, ValueError):
                continue
            if math.isnan(value):
                # Python's float() happily parses "nan" — a missing target
                # value in the curated CSV, not a plottable observation.
                continue
            date = row[date_idx]
            if not date:
                continue
            points.append((date, value))

        points.sort(key=lambda point: point[0])
        return points

    def _row_matches_key(
        self, row: list[str], key_indices: dict[str, int], key_values: dict[str, object]
    ) -> bool:
        # Single-series mode has no key columns at all — every row belongs
        # to the one series. Otherwise every key column's cell must match,
        # compared numerically when both sides parse as numbers (business
        # keys are frequently ids like `1` vs the CSV's `"1"`) and as plain
        # strings otherwise.
        for name, value in key_values.items():
            index = key_indices.get(name)
            if index is None:
                continue
            cell = row[index]
            if not self._values_match(cell, value):
                return False
        return True

    def _values_match(self, cell: str, value: object) -> bool:
        try:
            return float(cell) == float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return cell == str(value)

    # ------------------------------------------------------------------
    # Derived feature columns (Priority C)
    # ------------------------------------------------------------------

    def _with_derived_feature_columns(
        self, run_id: str, columns: list[str], rows: list[list[str]]
    ) -> tuple[list[str], list[list[str]]]:
        """Append this run's actual derived feature columns to the curated
        preview — computed with the same formulas `SupervisedTreeModel`
        uses (lag = shift, rolling mean = trailing rolling().mean(), month/
        quarter = date parts), grouped by business key so a lag or rolling
        mean is never computed across two different keys' rows.

        Reflects the run's own actual selection — read back from
        `run_metadata["derived_features"]`, never guessed — so two runs
        with different selections show different preview columns, and a
        run with none selected shows none added. `columns`/`rows` are
        returned unchanged (not even copied) whenever there is nothing to
        add, which is the common case (most runs, and every call after the
        underlying data has already been augmented once is avoided by the
        caller never re-augmenting a returned pair).
        """
        plan = self._derived_feature_plan(run_id, columns)
        if plan is None:
            return columns, rows
        date_column, target_column, group_columns, lags, windows, calendar = plan
        if not (lags or windows or calendar):
            return columns, rows

        frame = pd.DataFrame(rows, columns=columns)
        target = pd.to_numeric(frame[target_column], errors="coerce")
        dates = pd.to_datetime(frame[date_column], errors="coerce")

        # Sort once (by key, then date) so shift()/rolling() walk each
        # series in true chronological order, then every computed column is
        # re-aligned back to the file's own row order before being shown —
        # the preview must display rows in the same order the rest of the
        # page already assumes.
        sort_frame = pd.DataFrame({"__date__": dates, **{c: frame[c] for c in group_columns}})
        order = sort_frame.sort_values(["__date__", *group_columns] if not group_columns else [*group_columns, "__date__"]).index
        target_sorted = target.loc[order]
        group_series = [frame[c].loc[order] for c in group_columns]

        new_columns: dict[str, pd.Series] = {}
        for lag in lags:
            shifted = target_sorted.groupby(group_series).shift(lag) if group_columns else target_sorted.shift(lag)
            new_columns[f"lag_{lag}"] = shifted.reindex(frame.index)
        for window in windows:
            if group_columns:
                rolled = target_sorted.groupby(group_series).apply(
                    lambda s: s.shift(1).rolling(window).mean()
                )
                rolled.index = rolled.index.get_level_values(-1)
            else:
                rolled = target_sorted.shift(1).rolling(window).mean()
            new_columns[f"rolling_mean_{window}"] = rolled.reindex(frame.index)
        if "month" in calendar:
            new_columns["month"] = dates.dt.month
        if "quarter" in calendar:
            new_columns["quarter"] = dates.dt.quarter

        augmented_columns = columns + list(new_columns.keys())
        augmented_rows = [
            row + [self._cell_text(series.iloc[i]) for series in new_columns.values()]
            for i, row in enumerate(rows)
        ]
        return augmented_columns, augmented_rows

    def _cell_text(self, value: object) -> str:
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    def _derived_feature_plan(
        self, run_id: str, columns: list[str]
    ) -> tuple[str, str, list[str], list[int], list[int], list[str]] | None:
        """This run's date/target/key columns and its resolved derived
        feature selection — `None` if run metadata is unavailable or the
        configured columns are not actually present in the curated file.
        """
        try:
            result = self._executor.get_result(run_id)
        except Exception:  # noqa: BLE001 - an unreadable run simply has no preview additions
            return None
        metadata = result.run_metadata or {}
        configuration = metadata.get("configuration") or {}
        date_column = configuration.get("date_column")
        target_column = configuration.get("target_column")
        if not date_column or not target_column or date_column not in columns or target_column not in columns:
            return None
        key_columns = [c for c in (configuration.get("key_columns") or []) if c in columns]

        selected = metadata.get("derived_features")
        selected_ids = set(selected) if selected is not None else SUPPORTED_DERIVED_FEATURE_IDS
        lags = [n for n in SUPPORTED_LAGS if f"lag_{n}" in selected_ids]
        windows = [n for n in SUPPORTED_ROLLING_WINDOWS if f"rolling_mean_{n}" in selected_ids]
        calendar = [name for name in SUPPORTED_CALENDAR_FEATURES if name in selected_ids]
        return date_column, target_column, key_columns, lags, windows, calendar

    def _curated_uri(self, run_id: str) -> str | None:
        try:
            result = self._executor.get_result(run_id)
        except Exception:  # noqa: BLE001 - an unreadable run simply has no preview
            return None
        return (result.run_metadata or {}).get("curated_dataset_uri") or None


_service: DatasetPreviewService | None = None


def get_dataset_preview_service() -> DatasetPreviewService:
    global _service
    if _service is None:
        _service = DatasetPreviewService()
    return _service
