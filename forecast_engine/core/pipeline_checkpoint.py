"""Cross-task checkpointing for `PipelineContext` — what makes the
multi-task Databricks Serverless workflow possible (see
`databricks/resources/forecast_job_serverless.yml`).

Each task in that job is a separate process; nothing about a
`PipelineContext` survives between them except what is written to
storage. Rather than inventing a new artifact type per stage boundary,
this module pickles the *same* `PipelineContext` object that already
flows between stages in-process today — the minimum change needed to make
two adjacent stages independent Databricks tasks.

Local execution and the DCS job never call this: both still run the whole
pipeline as a single process via `ForecastEnginePipeline.run()`, exactly
as before this module existed.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from forecast_engine.core.pipeline_context import PipelineContext


def save_checkpoint(context: PipelineContext, path: str | Path) -> None:
    """Persist `context` for a later task to resume from.

    `on_stage_change` (a live-status callback bound to this process) is
    dropped before pickling — it is never valid in another process, and
    the task that loads this checkpoint re-attaches its own.
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
    """Fold a second parallel branch's contribution into `base`, in place.

    Used only where the DAG forks and rejoins: Persist Winning Models and
    Export Forecasts both descend from Rank & Select and each write to a
    disjoint slice of the context (`model_storage_results` and
    `forecast_export_result` respectively), so this only needs to copy
    over what `other` uniquely added — its own field(s), its stage
    record(s) (so the run's stage trail shows both branches), and any
    metadata key `base` does not already have.
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
