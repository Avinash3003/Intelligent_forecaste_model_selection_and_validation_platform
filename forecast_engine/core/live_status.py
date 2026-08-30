"""Interim progress, written after every stage transition.

The run summary is complete but only readable at the very end, so a caller
polling a live run would otherwise see every stage as Pending until the
whole thing finished. This writes the current stage trail to a small JSON
file instead, readable at any time.

Writes are atomic (temp file then replace), so a reader polling mid-write
never sees half a document.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forecast_engine.core import storage

if TYPE_CHECKING:
    from forecast_engine.core.pipeline_context import PipelineContext


class LiveStatusWriter:
    """Writes the stage trail on every transition.

    Never raises: a broken writer must not interrupt the run it reports on.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def __call__(self, context: "PipelineContext") -> None:
        payload: dict[str, Any] = {
            "run_id": context.run_id,
            "started_at": context.started_at.isoformat(timespec="seconds"),
            "stages": [stage.to_dict() for stage in context.stages],
        }
        try:
            self._write_atomic(payload)
        except OSError:
            pass

    def _write_atomic(self, payload: dict[str, Any]) -> None:
        # A UC Volume with no POSIX mount has no rename, so the tmp+replace
        # dance below cannot run there. It does not need to: the Files API
        # writes this payload in a single request (the SDK only switches to
        # multipart above files_ext_multipart_upload_min_stream_size, which
        # is 50 MiB — a status document is a few KB), and an object store
        # PUT replaces the object wholesale. A reader sees either the
        # previous complete document or the new one, never a splice of the
        # two. That is the same guarantee os.replace gives, reached by a
        # different mechanism, so neither route can serve a partial file.
        if not storage.supports_atomic_replace(self._path):
            storage.write_text(self._path, json.dumps(payload))
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Written into the same directory as the final destination so the
        # subsequent os.replace is a same-filesystem rename, which is what
        # makes it atomic — a cross-filesystem "rename" silently falls back
        # to copy+delete, which is exactly the non-atomic behaviour this
        # exists to avoid.
        fd, tmp_path = tempfile.mkstemp(dir=self._path.parent, prefix=".live_status_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle)
            os.replace(tmp_path, self._path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
