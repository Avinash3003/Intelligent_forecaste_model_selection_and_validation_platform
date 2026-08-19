"""Shows the curated dataset — what the models actually trained on.

Deliberately not the raw upload: the curated file is post-preprocessing, so
it is the same data the decision on this page was made from, and it is far
smaller (one row per key per month) so pagination stays cheap.

The file sits on local disk for local runs and in a UC volume for cloud
runs. Both are POSIX paths; only this process's ability to open them
differs, so an unreadable local path is fetched through the runner's
workspace client instead.
"""

from __future__ import annotations

import csv
import io
import math
import threading
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from app.config.settings import Settings, get_settings
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

    def __init__(
        self,
        executor: PipelineExecutor | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._executor = executor or get_pipeline_executor()
        # Read only to rebuild a durable volume path when the recorded one no
        # longer resolves; injectable so a test can vary the mode/roots.
        self._settings = settings or get_settings()
        self._cache: "OrderedDict[str, tuple[list[str], list[list[str]]]]" = OrderedDict()
        # This service is a process-wide singleton (get_dataset_preview_
        # service) and FastAPI runs sync path operations like get_preview()
        # in a threadpool, so concurrent requests for different runs/pages
        # genuinely interleave on this dict — a plain OrderedDict's
        # check-then-act sequences (move_to_end after an `in` check,
        # popitem after a len() check) are not atomic as a whole even
        # though CPython's GIL makes each individual op atomic.
        self._lock = threading.Lock()

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
        with self._lock:
            if run_id in self._cache:
                self._cache.move_to_end(run_id)
                return self._cache[run_id]

        uri = self._curated_uri(run_id)
        if not uri:
            return None

        text = self._read(uri, run_id)
        if text is None:
            return None

        reader = csv.reader(io.StringIO(text))
        try:
            columns = next(reader)
        except StopIteration:
            return [], []
        rows = list(reader)

        # The parse above happens outside the lock — it is pure CPU/IO work
        # on a local `text` string, not shared state, and holding a lock
        # across it would serialize every concurrent preview load for no
        # reason. Two threads racing to parse the same uncached run_id both
        # do the (idempotent) work once each; only the dict mutation below
        # needs to be exclusive.
        with self._lock:
            self._cache[run_id] = (columns, rows)
            self._cache.move_to_end(run_id)
            if len(self._cache) > _CACHE_CAPACITY:
                self._cache.popitem(last=False)
        return columns, rows

    def _read(self, uri: str, run_id: str | None = None) -> str | None:
        """The curated file, from the first location that actually yields it.

        A run records the absolute path its curated file was written to. That
        path is durable for a cloud run (a UC volume) and for a local run (the
        API host's own disk) — but a path recorded by an *earlier* deployment
        can be stale: App Service runs the app from a per-build /tmp
        directory, so an absolute path captured under one build no longer
        exists under the next.

        Rather than fail, each candidate below is tried in turn: the recorded
        path first (the common case, and the only one local runs use), then —
        for cloud runs — the path the same file would occupy in the configured
        curated volume today. That makes resolution survive a restart, a
        redeploy and a new process, without changing where anything is
        written.
        """
        for candidate in self._candidate_uris(uri, run_id):
            path = Path(candidate)
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
            contents = self._read_remote(candidate)
            if contents is not None:
                return contents
        return None

    def _candidate_uris(self, uri: str, run_id: str | None) -> list[str]:
        """The recorded location, then the durable one it maps to today.

        Local execution gets exactly one candidate — its recorded path is the
        authoritative location and there is no volume to fall back to, so
        local behaviour is unchanged.
        """
        candidates = [uri]

        mode = (self._settings.execution_mode or "local").strip().lower()
        if mode != "databricks" or not run_id:
            return candidates

        # The curated writer lays files out as <root>/runs/<run_id>/<file>
        # (see DatabricksRunner._curated_root), so the filename plus the
        # configured root is enough to rebuild today's location. Nothing here
        # is hardcoded to a workspace: the root comes from settings.
        root = (self._settings.databricks_curated_volumes_root or "").rstrip("/")
        filename = Path(uri).name
        if root and filename:
            rebuilt = f"{root}/runs/{run_id}/{filename}"
            if rebuilt != uri:
                candidates.append(rebuilt)
        return candidates

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
        """Every (date, target) observation for one business key.

        Reuses the cached curated-file parse get_preview() already does.
        Returns None (not []) when the curated file is unavailable, so a
        caller can tell "no curated data" from "no rows for this key".
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
        """Append this run's derived feature columns to the preview.

        Uses the same formulas the tree models do (lag = shift, rolling mean
        = trailing mean, month/quarter = date parts), grouped by business key
        so a lag never spans two keys.

        Reads the run's real selection from run_metadata, never a guess, so
        two runs with different selections preview different columns.
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
