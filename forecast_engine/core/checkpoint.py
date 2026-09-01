"""Cross-process state handoff for the Databricks multi-task workflow.

Each pipeline phase runs as its own Databricks task — a separate process,
sometimes a separate container. Nothing in memory survives between them, so
the driver checkpoints a shrunk PipelineContext after every phase, and the
next task's process loads it back before running its own phase.

Never a mechanism for passing large objects through task *parameters* —
the checkpoint travels through the same artifacts storage every other run
output already uses, keyed by run_id like everything else there.
"""

from __future__ import annotations

import pickle
from dataclasses import replace
from typing import Any

from forecast_engine.core import storage
from forecast_engine.core.pipeline_context import PipelineContext

CHECKPOINT_FILENAME = "checkpoint.pkl"


def checkpoint_uri(artifacts_root: str, run_id: str) -> str:
    return f"{artifacts_root.rstrip('/')}/{run_id}/{CHECKPOINT_FILENAME}"


def save(context: PipelineContext, artifacts_root: str) -> None:
    """Persist this run's state for the next task to resume.

    Excludes what must never cross a process boundary this way: the raw and
    prepared DataFrames (large, and every later phase reads
    curated_dataset_uri/series instead — see run_pipeline._load_curated_dataset),
    the live-status callback (a local file handle, not a value), and the
    key-execution executor's live Ray objects (invalid once this process's
    Ray cluster shuts down) — its plain per-key state travels separately via
    StagedKeyExecution.snapshot().
    """
    executor = context.key_stage_executor
    snapshot = executor.snapshot() if executor is not None else None

    shrunk = replace(
        context,
        raw_dataset=None,
        prepared_dataset=None,
        key_stage_executor=None,
        on_stage_change=None,
    )
    payload = pickle.dumps(
        {"context": shrunk, "key_execution_snapshot": snapshot}, protocol=pickle.HIGHEST_PROTOCOL
    )
    storage.write_bytes(checkpoint_uri(artifacts_root, context.run_id), payload)


def load(artifacts_root: str, run_id: str) -> tuple[PipelineContext, dict[str, Any] | None]:
    """The prior task's context and key-execution snapshot (or None).

    Raises FileNotFoundError when no checkpoint exists — a task must fail
    clearly when its required prior-stage output was never written, not
    silently start from nothing.
    """
    uri = checkpoint_uri(artifacts_root, run_id)
    try:
        payload = storage.read_bytes(uri)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No checkpoint found for run '{run_id}' at {uri}. The previous pipeline stage may not have completed."
        ) from None

    data = pickle.loads(payload)
    return data["context"], data.get("key_execution_snapshot")
