"""The multi-task Databricks Serverless workflow (Priority #2).

`ForecastEnginePipeline.run_stage()` runs one stage group per Databricks
task, checkpointing `PipelineContext` between them via
`forecast_engine.core.pipeline_checkpoint` — see
`databricks/resources/forecast_job_serverless.yml` for the actual task DAG.
These tests cover the refactor itself, not any pipeline stage's own logic
(that is already covered elsewhere): that `STAGE_GROUPS` still runs the
exact same stages `run()` does, that checkpointing round-trips a context,
that the one fork/join (Persist Models / Export Forecasts) merges
correctly, and that running the whole pipeline task-by-task produces the
same result as running it in one process.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from forecast_engine.core.forecast_configuration import ForecastConfiguration
from forecast_engine.core.pipeline_checkpoint import load_checkpoint, merge_checkpoints, save_checkpoint
from forecast_engine.core.pipeline_context import PipelineContext, StageRecord
from forecast_engine.run_pipeline import ForecastEnginePipeline


# ---------------------------------------------------------------------
# STAGE_GROUPS structure
# ---------------------------------------------------------------------

_RUN_STAGE_ORDER = (
    "_load_dataset",
    "_detect_frequency",
    "_assess_quality",
    "_preprocess",
    "_persist_curated_dataset",
    "_verify_curated_dataset",
    "_generate_groups",
    "_build_series",
    "_train_models",
    "_evaluate_models",
    "_generate_explainability",
    "_select_production_models",
    "_persist_winning_models",
    "_export_forecasts",
    "_generate_business_insights",
    "_mirror_artifacts",
    "_track_to_mlflow",
)


def test_stage_groups_cover_exactly_the_stages_run_calls_in_order():
    flattened = tuple(
        method for group in ForecastEnginePipeline.STAGE_GROUPS.values() for method in group
    )
    assert flattened == _RUN_STAGE_ORDER


def test_every_stage_group_has_a_predecessor_entry():
    assert set(ForecastEnginePipeline.STAGE_GROUPS) == set(ForecastEnginePipeline.STAGE_GROUP_PREDECESSORS)


def test_only_business_insights_has_two_predecessors():
    multi = {
        group: preds
        for group, preds in ForecastEnginePipeline.STAGE_GROUP_PREDECESSORS.items()
        if len(preds) > 1
    }
    assert multi == {"business_insights": ("persist_models", "export_forecasts")}


def test_persist_models_and_export_forecasts_both_descend_from_rank_select():
    preds = ForecastEnginePipeline.STAGE_GROUP_PREDECESSORS
    assert preds["persist_models"] == ("rank_select",)
    assert preds["export_forecasts"] == ("rank_select",)


def test_mlflow_tracking_is_last_and_has_no_successor_reading_it():
    order = list(ForecastEnginePipeline.STAGE_GROUP_PREDECESSORS)
    assert order[-1] == "mlflow_tracking"


# ---------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------


def _context() -> PipelineContext:
    return PipelineContext(
        run_id="run-1",
        dataset_path="d.csv",
        configuration=ForecastConfiguration(date_column="date", target_column="sales"),
    )


def test_a_checkpoint_round_trips_the_context(tmp_path):
    ctx = _context()
    ctx.frequency = "Monthly"
    ctx.metadata["raw_rows"] = 42

    save_checkpoint(ctx, tmp_path / "after_load_prepare.pkl")
    restored = load_checkpoint(tmp_path / "after_load_prepare.pkl")

    assert restored.run_id == "run-1"
    assert restored.frequency == "Monthly"
    assert restored.metadata["raw_rows"] == 42


def test_saving_drops_the_live_status_callback(tmp_path):
    ctx = _context()
    ctx.on_stage_change = lambda context: None

    save_checkpoint(ctx, tmp_path / "ctx.pkl")
    restored = load_checkpoint(tmp_path / "ctx.pkl")

    assert restored.on_stage_change is None


def test_save_checkpoint_creates_missing_parent_directories(tmp_path):
    save_checkpoint(_context(), tmp_path / "nested" / "dir" / "ctx.pkl")
    assert (tmp_path / "nested" / "dir" / "ctx.pkl").exists()


# ---------------------------------------------------------------------
# merge_checkpoints — the fork/join
# ---------------------------------------------------------------------


def test_merge_copies_the_other_branchs_unique_field():
    base = _context()
    base.model_storage_results = [{"forecast_group": "1 | 1", "persisted": True}]
    other = _context()
    other.forecast_export_result = {"persisted": True, "rows": 12}

    merge_checkpoints(base, other)

    assert base.model_storage_results == [{"forecast_group": "1 | 1", "persisted": True}]
    assert base.forecast_export_result == {"persisted": True, "rows": 12}


def test_merge_never_clobbers_bases_own_field_with_an_empty_one():
    base = _context()
    base.forecast_export_result = {"persisted": True, "rows": 12}
    other = _context()  # never ran Export Forecasts; forecast_export_result is the default {}

    merge_checkpoints(base, other)

    assert base.forecast_export_result == {"persisted": True, "rows": 12}


def test_merge_appends_the_other_branchs_stage_record_once():
    base = _context()
    base.stages = [StageRecord(name="Rank & Select"), StageRecord(name="Persist Models")]
    other = _context()
    other.stages = [StageRecord(name="Rank & Select"), StageRecord(name="Export Forecasts")]

    merge_checkpoints(base, other)

    names = [stage.name for stage in base.stages]
    assert names == ["Rank & Select", "Persist Models", "Export Forecasts"]


def test_merge_unions_metadata_without_overwriting_bases_keys():
    base = _context()
    base.metadata["models_persisted"] = 1
    other = _context()
    other.metadata["models_persisted"] = 999  # would only happen if merge ran the wrong way
    other.metadata["some_export_fact"] = "x"

    merge_checkpoints(base, other)

    assert base.metadata["models_persisted"] == 1
    assert base.metadata["some_export_fact"] == "x"


# ---------------------------------------------------------------------
# MLflowTrackingPipeline.resume()
# ---------------------------------------------------------------------


@dataclass
class _FakeClient:
    configured: bool = False
    started_with: dict | None = None
    available: bool = True

    def is_available(self):
        return self.available

    def configure(self):
        self.configured = True

    def start_run(self, run_name, tags=None, resume_run_id=None):
        self.started_with = {"run_name": run_name, "tags": tags, "resume_run_id": resume_run_id}
        return object()


def _tracking_pipeline(client):
    from forecast_engine.config.mlflow_config import MLflowConfig
    from forecast_engine.s12_tracking.tracking_pipeline import MLflowTrackingPipeline

    return MLflowTrackingPipeline(MLflowConfig(), client)


def test_resume_reopens_the_run_by_id():
    client = _FakeClient()
    pipeline = _tracking_pipeline(client)

    pipeline.resume("mlflow-run-abc")

    assert client.started_with["resume_run_id"] == "mlflow-run-abc"
    assert pipeline._open is True


def test_resume_is_a_no_op_for_a_falsy_run_id():
    client = _FakeClient()
    pipeline = _tracking_pipeline(client)

    pipeline.resume(None)

    assert client.started_with is None
    assert pipeline._open is False


def test_resume_is_a_no_op_when_the_client_is_unavailable():
    client = _FakeClient(available=False)
    pipeline = _tracking_pipeline(client)

    pipeline.resume("mlflow-run-abc")

    assert client.started_with is None
    assert pipeline._open is False


def test_resume_never_raises_when_the_reopen_fails():
    class _RaisingClient(_FakeClient):
        def start_run(self, run_name, tags=None, resume_run_id=None):
            raise RuntimeError("workspace unreachable")

    pipeline = _tracking_pipeline(_RaisingClient())

    pipeline.resume("mlflow-run-abc")  # must not raise

    assert pipeline._open is False


# ---------------------------------------------------------------------
# End-to-end equivalence: run_stage() task-by-task vs. run() in one process
# ---------------------------------------------------------------------


def _synthetic_dataset(tmp_path) -> str:
    # 30 months of a simple trending, mildly seasonal series — enough
    # history for ARIMA's default min_observations (20) with room to
    # spare, small enough that both pipeline runs stay fast.
    dates = pd.date_range("2022-01-01", periods=30, freq="MS")
    values = [100 + i * 2 + (5 if i % 12 in (10, 11) else 0) for i in range(30)]
    frame = pd.DataFrame({"date": dates, "sales": values})
    path = tmp_path / "sales.csv"
    frame.to_csv(path, index=False)
    return str(path)


def _configuration() -> ForecastConfiguration:
    return ForecastConfiguration(date_column="date", target_column="sales")


@pytest.fixture
def _pipeline_kwargs(tmp_path):
    from forecast_engine.config.mlflow_config import MLflowConfig
    from forecast_engine.config.pipeline_config import (
        ArtifactsMirrorConfig,
        CuratedStorageConfig,
        ForecastExportConfig,
        ModelStorageConfig,
        PipelineConfig,
    )

    # Local, disposable storage roots — same shape a real run uses, just
    # confined to tmp_path so the test leaves nothing behind. MLflow is
    # explicitly disabled: this test is about the checkpoint/task-split
    # refactor, not tracking, and a real local mlruns/mlflow.db would be
    # slow and leave behind side effects the pipeline's own tests don't
    # need. `resume()` has its own dedicated unit tests above.
    return {
        "pipeline_config": PipelineConfig(
            curated_storage=CuratedStorageConfig(root_dir=str(tmp_path / "curated")),
            model_storage=ModelStorageConfig(root_dir=str(tmp_path / "models")),
            forecast_export=ForecastExportConfig(root_dir=str(tmp_path / "forecasts")),
            artifacts_mirror=ArtifactsMirrorConfig(root_dir=str(tmp_path / "artifacts")),
        ),
        "mlflow_config": MLflowConfig(enabled=False),
    }


def test_running_task_by_task_matches_running_in_one_process(tmp_path, _pipeline_kwargs):
    dataset_path = _synthetic_dataset(tmp_path)

    baseline_pipeline = ForecastEnginePipeline(**_pipeline_kwargs)
    baseline = baseline_pipeline.run(
        dataset_path, _configuration(), run_id="equivalence-baseline", selected_models=["arima"]
    )

    staged_pipeline = ForecastEnginePipeline(**_pipeline_kwargs)
    checkpoint_dir = tmp_path / "checkpoints"
    context = None
    for stage_group in ForecastEnginePipeline.STAGE_GROUPS:
        context = staged_pipeline.run_stage(
            stage_group,
            checkpoint_dir=checkpoint_dir,
            dataset_path=dataset_path,
            configuration=_configuration(),
            run_id="equivalence-staged",
            selected_models=["arima"],
        )

    baseline_summary = baseline.summary()
    staged_summary = context.summary()

    _TIMING_KEYS = {"trained_at", "started_at", "completed_at"}

    def _is_timing_key(key: str) -> bool:
        return key in _TIMING_KEYS or key.endswith("_seconds")

    def _without_timing(value):
        # Wall-clock duration and timestamps are inherently
        # non-deterministic between two separate executions, at any
        # nesting depth (e.g. per-record metadata); every other field is.
        if isinstance(value, dict):
            return {k: _without_timing(v) for k, v in value.items() if not _is_timing_key(k)}
        if isinstance(value, list):
            return [_without_timing(v) for v in value]
        return value

    # run_id differs by construction (each run was given its own); every
    # other deterministic field — what was trained, ranked, selected and
    # forecast — must match exactly between the two execution paths.
    assert staged_summary["group_count"] == baseline_summary["group_count"]
    assert _without_timing(staged_summary["training_report"]) == _without_timing(baseline_summary["training_report"])
    assert _without_timing(staged_summary["evaluation_report"]) == _without_timing(baseline_summary["evaluation_report"])
    assert _without_timing(staged_summary["ranking_report"]) == _without_timing(baseline_summary["ranking_report"])

    baseline_selection = _without_timing(baseline_summary["production_selection_report"])
    staged_selection = _without_timing(staged_summary["production_selection_report"])
    assert staged_selection == baseline_selection

    def _normalize_run_id(value):
        # Written paths embed the run_id, which intentionally differs
        # between the two runs (each was given its own) — normalize it out
        # so this only checks the fork/join actually contributed both
        # halves, not just one branch.
        if isinstance(value, str):
            return value.replace("equivalence-staged", "RUN_ID").replace("equivalence-baseline", "RUN_ID")
        if isinstance(value, dict):
            return {k: _normalize_run_id(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_normalize_run_id(v) for v in value]
        return value

    assert _normalize_run_id(staged_summary["model_storage_results"]) == _normalize_run_id(
        baseline_summary["model_storage_results"]
    )
    assert _normalize_run_id(staged_summary["forecast_export_result"]) == _normalize_run_id(
        baseline_summary["forecast_export_result"]
    )

    # Every stage the single-process run recorded is present in the
    # task-by-task run's merged trail, including both fork branches.
    baseline_stage_names = {stage["name"] for stage in baseline_summary["stages"]}
    staged_stage_names = {stage["name"] for stage in staged_summary["stages"]}
    assert staged_stage_names == baseline_stage_names
    assert "Persist Models" in staged_stage_names
    assert "Export Forecasts" in staged_stage_names


def test_a_task_failure_leaves_no_checkpoint_for_the_next_task(tmp_path, _pipeline_kwargs):
    from forecast_engine.utils.exceptions import ForecastEngineError

    pipeline = ForecastEnginePipeline(**_pipeline_kwargs)
    checkpoint_dir = tmp_path / "checkpoints"
    missing_dataset = str(tmp_path / "does_not_exist.csv")

    with pytest.raises(ForecastEngineError):
        pipeline.run_stage(
            "load_prepare",
            checkpoint_dir=checkpoint_dir,
            dataset_path=missing_dataset,
            configuration=_configuration(),
            run_id="failure-test",
        )

    assert not (checkpoint_dir / "after_load_prepare.pkl").exists()
