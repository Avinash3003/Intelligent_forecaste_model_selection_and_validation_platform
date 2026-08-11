"""Artifacts mirror — a blob-accessible copy of this run's business
insights and LLM trace, alongside MLflow's own copy.

MLflow remains the authoritative record (Section 6.13) — this writer reads
the same already-serialized data `s12_tracking/artifact_logger.py` logs to
MLflow and writes an identical copy here, purely so the content is
reachable by direct blob access without going through the Databricks
MLflow UI. Nothing is recomputed, re-rendered, or duplicated in a way that
could disagree with MLflow's copy: both read the same `PipelineResult`
fields, and only the destination differs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forecast_engine.config.pipeline_config import ArtifactsMirrorConfig
from forecast_engine.s03_storage.model_writer import sanitize_forecast_key


class ArtifactsMirrorWriter:
    """Writes a run's business insights and LLM trace outside MLflow."""

    def __init__(self, config: ArtifactsMirrorConfig | None = None) -> None:
        self._config = config or ArtifactsMirrorConfig()

    def write(self, business_insights: dict[str, Any], llm_trace: dict[str, Any], run_id: str) -> dict[str, Any]:
        """Persist the two artifacts already produced for this run.

        Never raises: a run's insights are already complete and correct
        by the time this runs, so a mirroring failure is reported on its
        own record, never allowed to fail the run.
        """
        if not self._config.enabled:
            return {"enabled": False, "persisted": []}

        run_dir = Path(self._config.root_dir) / sanitize_forecast_key(run_id)
        persisted: list[dict[str, Any]] = []

        if business_insights:
            persisted.append(self._write_one(run_dir, "business_insights.json", business_insights))
        if llm_trace and llm_trace.get("calls"):
            persisted.append(self._write_one(run_dir, "llm_trace.json", llm_trace))

        return {"enabled": True, "persisted": persisted}

    def _write_one(self, run_dir: Path, filename: str, data: dict[str, Any]) -> dict[str, Any]:
        path = run_dir / filename
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - one file must not block another
            return {"file": filename, "persisted": False, "uri": None, "error": str(exc)}
        return {"file": filename, "persisted": True, "uri": str(path), "error": None}
