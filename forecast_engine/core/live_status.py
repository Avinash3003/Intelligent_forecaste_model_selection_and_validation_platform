"""Live execution status — the engine's *interim* progress report.

`PipelineContext.summary()` is the run's complete record, but it is only
ever read once, at the very end (`run_pipeline.py`'s `--summary-out`). A
caller polling a still-executing run has nothing to read until then, which
is why status pages used to show every stage as "Pending" right up until
the moment the whole run finished.

`LiveStatusWriter` closes that gap: `PipelineContext.on_stage_change`
(core/pipeline_context.py) calls it after every stage transition, and it
writes the current stage trail to a small JSON file a caller can read at
any time — including mid-stage, which is what lets "Load Dataset" show as
Running instead of Pending while it is actually running.

Writes are atomic (write to a temp file, then `os.replace`) so a reader
polling the file mid-write never sees a half-written JSON document.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forecast_engine.core.pipeline_context import PipelineContext


class LiveStatusWriter:
    """Writes `PipelineContext`'s current stage trail to `path` on every
    stage transition. Never raises — a broken writer (e.g. a deleted
    parent directory) must not interrupt the forecasting run it is only
    reporting on.
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
