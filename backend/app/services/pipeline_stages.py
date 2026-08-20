"""The pipeline's stage vocabulary — one set of names for every surface.

The engine, the Databricks Serverless job and the UI all used to name the
same seventeen stages differently ("Generate Explainability (SHAP)" vs
`explain_models` vs a third wording in the deck), which made a run's trail
impossible to follow across surfaces. They now share one vocabulary:

    every label here is the Title Case of its Databricks `task_key`

so `explain_models` reports "Explain Models" and `mlflow_tracking` reports
"MLflow Tracking". Nine of the eleven Databricks tasks are exactly 1:1 with
a stage; only `load_prepare` and `build_series` fan out to several, and
those keep the same two-word shape so the trail reads uniformly.

Defined here rather than in deployment_service so estimation (which only
needs to translate a stage name) does not have to import the executor
stack to get it.

Sources kept in sync, verified by tests/backend/test_stage_trail.py:
  - forecast_engine/run_pipeline.py     `begin_stage(...)` calls
  - databricks/resources/forecast_job_serverless.yml   `task_key`s
"""

from __future__ import annotations

# The complete set, in execution order. Both the shape rendered for a run
# that has not reported a trail yet AND the skeleton a live run's reported
# stages are merged onto, so the UI always shows the whole pipeline with
# stages not yet reached marked Pending rather than silently omitted.
#
# Also the progress denominator, so a missing entry here overstates how far
# along every run is.
PIPELINE_STAGES = [
    "Load Dataset",
    "Detect Frequency",
    "Assess Quality",
    "Preprocess Dataset",
    "Persist Curated",
    "Verify Curated",
    "Generate Groups",
    "Build Series",
    "Train Models",
    "Evaluate Models",
    "Explain Models",
    "Rank & Select",
    "Persist Models",
    "Export Forecasts",
    "Business Insights",
    "Mirror Artifacts",
    "MLflow Tracking",
]

# Stage names used before the vocabulary was unified, mapped onto it. Runs
# completed under the old names are still on disk (MLflow artifacts,
# live-status files), and their trails must keep rendering against the
# current skeleton instead of being appended as seventeen extra unknown
# stages below seventeen Pending ones. Translation happens on read only —
# nothing rewrites a stored artifact.
_LEGACY_STAGE_NAMES = {
    "Assess Data Quality": "Assess Quality",
    "Persist Curated Dataset": "Persist Curated",
    "Verify Curated Dataset": "Verify Curated",
    "Generate Forecast Groups": "Generate Groups",
    "Build Forecast Series": "Build Series",
    "Generate Explainability (SHAP)": "Explain Models",
    "Rank & Select Production Models": "Rank & Select",
    "Persist Winning Models": "Persist Models",
    "Generate Business Insights": "Business Insights",
    "Track to MLflow": "MLflow Tracking",
}


def canonical_stage_name(name: str) -> str:
    """The current label for a stage name, old or new.

    Identity for names already current, so it is safe to call on every
    stage unconditionally rather than branching on which era produced it.
    """
    return _LEGACY_STAGE_NAMES.get(name, name)


# ----------------------------------------------------------------------
# Display phases
# ----------------------------------------------------------------------
#
# PIPELINE_STAGES above is the ENGINE's contract — seventeen stages, kept in
# lockstep with `begin_stage(...)` and verified by tests. It is the right
# granularity for a checkpoint boundary and the wrong one for a person: a
# seventeen-row trail is hard to talk through, and most rows are sub-second
# bookkeeping nobody needs to see.
#
# These seven phases are a VIEW over those stages, for the UI only. Nothing
# reports against them and no engine or Databricks task is renamed to match —
# the DAG keeps its own task_keys, deliberately, because the two serve
# different audiences.
#
# Every one of the seventeen stages belongs to exactly one phase, checked by
# tests, so a stage can never quietly vanish from the trail.
PIPELINE_PHASES: list[tuple[str, tuple[str, ...]]] = [
    ("Load & Prepare", (
        "Load Dataset",
        "Detect Frequency",
        "Assess Quality",
        "Preprocess Dataset",
        "Persist Curated",
        "Verify Curated",
    )),
    ("Build Series", ("Generate Groups", "Build Series")),
    ("Train Models", ("Train Models",)),
    ("Evaluate Models", ("Evaluate Models",)),
    ("Explain Models", ("Explain Models",)),
    ("Rank & Select", ("Rank & Select",)),
    ("Publish Results", (
        "Persist Models",
        "Export Forecasts",
        "Business Insights",
        "Mirror Artifacts",
        "MLflow Tracking",
    )),
]

PHASE_LABELS: list[str] = [label for label, _ in PIPELINE_PHASES]

# Which phase a given engine stage rolls up into.
STAGE_TO_PHASE: dict[str, str] = {
    stage: label for label, stages in PIPELINE_PHASES for stage in stages
}
