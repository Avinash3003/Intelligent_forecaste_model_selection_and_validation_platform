"""The Databricks multi-task handoff: each phase checkpoints its state so
the next phase can resume in a brand new process, with no in-memory state
and no live Ray objects surviving the boundary.

test_full_pipeline_via_seven_checkpointed_stage_calls_matches_direct_run is
the real proof: it drives every phase through a FRESH ForecastEnginePipeline
instance each time (never reusing the previous phase's in-memory pipeline
or context), exactly like seven separate Databricks tasks would, and checks
the result against a normal single-process run().
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecast_engine.config.mlflow_config import MLflowConfig
from forecast_engine.config.model_config import ModelConfig
from forecast_engine.config.pipeline_config import (
    ArtifactsMirrorConfig,
    CuratedStorageConfig,
    ForecastExportConfig,
    ModelStorageConfig,
    PipelineConfig,
)
from forecast_engine.core import checkpoint
from forecast_engine.core.forecast_configuration import ForecastConfiguration
from forecast_engine.core.pipeline_context import PipelineContext
from forecast_engine.parallel.ray_executor import StagedKeyExecution, ray_available
from forecast_engine.run_pipeline import PHASE_ORDER, ForecastEnginePipeline

requires_ray = pytest.mark.skipif(not ray_available(), reason="Ray is not installed")


def _dataset(tmp_path, keys: int = 3, months: int = 36):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2019-01-01", periods=months, freq="MS")
    rows = []
    for k in range(1, keys + 1):
        values = 200 + 20 * k + np.linspace(0, 40, months) + rng.normal(0, 5, months)
        rows.extend({"date": d, "store": f"S{k}", "sales": round(float(v), 2)} for d, v in zip(dates, values))
    path = tmp_path / "sales.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _pipeline(tmp_path, *, parallel_keys: bool, mlflow_config: MLflowConfig | None = None) -> ForecastEnginePipeline:
    default = ModelConfig.default()
    registry = tuple(
        replace(spec, enabled=True) if spec.name == "seasonal_naive" else spec for spec in default.registry
    )
    return ForecastEnginePipeline(
        model_config=replace(default, registry=registry),
        mlflow_config=mlflow_config or MLflowConfig(enabled=False),
        pipeline_config=PipelineConfig(
            curated_storage=CuratedStorageConfig(root_dir=str(tmp_path / "curated")),
            model_storage=ModelStorageConfig(root_dir=str(tmp_path / "models")),
            forecast_export=ForecastExportConfig(root_dir=str(tmp_path / "forecasts")),
            artifacts_mirror=ArtifactsMirrorConfig(root_dir=str(tmp_path / "artifacts")),
        ),
        parallel_keys=parallel_keys,
    )


def test_snapshot_and_resume_round_trip_preserves_stage_state(tmp_path):
    dataset_path = _dataset(tmp_path, keys=2)
    pipeline = _pipeline(tmp_path, parallel_keys=True)
    context = PipelineContext.create(
        dataset_path=dataset_path,
        configuration=ForecastConfiguration(date_column="date", target_column="sales", key_columns=("store",)),
        run_id="snapshot-resume-test",
    )
    context.selected_models = ["seasonal_naive"]
    pipeline._load_dataset(context)
    pipeline._detect_frequency(context)
    pipeline._assess_quality(context)
    pipeline._preprocess(context)
    pipeline._generate_groups(context)
    pipeline._build_series(context)

    original = pipeline._stage_executor(context)
    original.run_training()

    snapshot = original.snapshot()
    # A fresh instance, exactly as a new Databricks task's own process
    # would build one — never the original object.
    resumed = StagedKeyExecution.resume(pipeline._key_workflow_config(context), snapshot)
    assert resumed is not original

    resumed_report, _ = resumed.run_evaluation()
    original_report, _ = original.run_evaluation()
    assert {r.group_id for r in resumed_report.results} == {r.group_id for r in original_report.results}
    assert resumed.failed_keys == original.failed_keys


def test_checkpoint_save_and_load_round_trip_excludes_live_state(tmp_path):
    dataset_path = _dataset(tmp_path, keys=1)
    context = PipelineContext.create(
        dataset_path=dataset_path,
        configuration=ForecastConfiguration(date_column="date", target_column="sales", key_columns=("store",)),
        run_id="checkpoint-round-trip-test",
    )
    context.raw_dataset = pd.DataFrame({"a": [1]})
    context.prepared_dataset = pd.DataFrame({"a": [1]})
    context.on_stage_change = lambda ctx: None
    context.begin_stage("Load Dataset")
    context.metadata["raw_rows"] = 10

    artifacts_root = str(tmp_path / "artifacts")
    checkpoint.save(context, artifacts_root)

    restored, snapshot = checkpoint.load(artifacts_root, context.run_id)
    assert restored is not context
    assert restored.run_id == context.run_id
    assert restored.metadata == {"raw_rows": 10}
    assert len(restored.stages) == 1
    assert restored.raw_dataset is None
    assert restored.prepared_dataset is None
    assert restored.on_stage_change is None
    assert snapshot is None  # no key_stage_executor was ever set on this context


def test_checkpoint_load_missing_run_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError):
        checkpoint.load(str(tmp_path / "artifacts"), "no-such-run")


@requires_ray
def test_full_pipeline_via_seven_checkpointed_stage_calls_matches_direct_run(tmp_path):
    dataset_path = _dataset(tmp_path, keys=3)
    config = ForecastConfiguration(date_column="date", target_column="sales", key_columns=("store",))

    direct = _pipeline(tmp_path / "direct", parallel_keys=True).run(
        str(dataset_path), config, run_id="checkpoint-direct-run", selected_models=["seasonal_naive"]
    )

    staged_root = tmp_path / "staged"
    artifacts_root = str(staged_root / "artifacts")
    run_id = "checkpoint-staged-run"

    context = None
    for stage in PHASE_ORDER:
        # A brand new pipeline instance every phase — nothing but what
        # checkpoint.save wrote for the last one crosses into this one, the
        # closest a single test process can get to seven separate tasks.
        pipeline = _pipeline(staged_root, parallel_keys=True)
        context = pipeline.run_checkpointed_stage(
            stage,
            run_id=run_id,
            dataset_path=str(dataset_path),
            configuration=config,
            artifacts_root=artifacts_root,
            selected_models=["seasonal_naive"],
        )

    assert context is not None
    assert all(s.status == "Completed" for s in context.stages)

    direct_winners = {r.group_id: r.model_name for r in direct.production_selection_report.results}
    staged_winners = {r.group_id: r.model_name for r in context.production_selection_report.results}
    assert staged_winners == direct_winners
    assert set(staged_winners) == {"S1", "S2", "S3"}


@requires_ray
def test_mlflow_tracking_survives_all_seven_checkpointed_tasks(tmp_path):
    """begin() opens the MLflow run in task 1's process only — every later
    task is a fresh process with no run in its own fluent state. Without
    resume() in run_checkpointed_stage, track() at publish_results reports
    "no run was open" even though a real run exists on the tracking server."""
    dataset_path = _dataset(tmp_path, keys=2)
    config = ForecastConfiguration(date_column="date", target_column="sales", key_columns=("store",))
    mlflow_config = MLflowConfig(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")

    staged_root = tmp_path / "staged"
    artifacts_root = str(staged_root / "artifacts")
    run_id = "checkpoint-mlflow-run"

    import mlflow

    context = None
    for stage in PHASE_ORDER:
        pipeline = _pipeline(staged_root, parallel_keys=True, mlflow_config=mlflow_config)
        context = pipeline.run_checkpointed_stage(
            stage,
            run_id=run_id,
            dataset_path=str(dataset_path),
            configuration=config,
            artifacts_root=artifacts_root,
            selected_models=["seasonal_naive"],
        )
        # A real task's process just exits, leaving nothing fluent-active
        # in the next one. One test process has no such boundary, so this
        # clears MLflow's own global active-run stack to match it — the
        # thing this test actually exercises, resume() reopening a run by
        # id, behaves identically either way.
        mlflow.end_run()

    assert context is not None
    assert context.tracking_result.logged is True, context.tracking_result.error
    assert context.tracking_result.status in ("logged", "logged_with_artifact_errors")
    assert context.tracking_result.models_registered >= 1
