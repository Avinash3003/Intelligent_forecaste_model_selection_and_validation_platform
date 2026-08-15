"""Lets the multi-task Databricks workflow hand a PipelineContext between tasks.

Each task is a separate process, so nothing survives between them but what
is written to storage. This pickles the same context object that already
flows between stages in-process, rather than inventing a per-boundary
artifact type.

Single-process runs never call this.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from forecast_engine.core.pipeline_context import PipelineContext


def save_checkpoint(context: PipelineContext, path: str | Path) -> None:
    """Persist the context for a later task.

    The live-status callback is dropped first — it is bound to this process,
    and the loading task attaches its own.
    """
    context.on_stage_change = None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(context, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str | Path) -> PipelineContext:
    """Restore a context an earlier task saved with `save_checkpoint()`."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def merge_checkpoints(base: PipelineContext, other: PipelineContext) -> None:
    """Fold a parallel branch's contribution into base, in place.

    Used only where the DAG forks and rejoins. The two branches write to
    disjoint slices of the context, so this copies only what the other side
    uniquely added: its fields, its stage records, and any new metadata.
    """
    if other.model_storage_results:
        base.model_storage_results = other.model_storage_results
    if other.forecast_export_result:
        base.forecast_export_result = other.forecast_export_result

    known_stage_names = {stage.name for stage in base.stages}
    for stage in other.stages:
        if stage.name not in known_stage_names:
            base.stages.append(stage)

    for key, value in other.metadata.items():
        base.metadata.setdefault(key, value)
