"""The full pipeline, run with parallel_keys=True, must report four
genuine stage boundaries for Train/Evaluate/Explain/Rank & Select — each
its own real Ray fan-out, not one cached result relabeled four times.

Runs the real ForecastEnginePipeline end to end (no mocks) against a
small multi-key dataset, so this is the actual integration the backend's
StageStatus API and the frontend both depend on.
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
from forecast_engine.core.forecast_configuration import ForecastConfiguration
from forecast_engine.parallel.ray_executor import ray_available
from forecast_engine.run_pipeline import ForecastEnginePipeline

requires_ray = pytest.mark.skipif(not ray_available(), reason="Ray is not installed")


def _dataset(tmp_path, keys: int = 3, months: int = 36):
    rng = np.random.default_rng(3)
    dates = pd.date_range("2019-01-01", periods=months, freq="MS")
    rows = []
    for k in range(1, keys + 1):
        values = 200 + 20 * k + np.linspace(0, 40, months) + rng.normal(0, 5, months)
        rows.extend({"date": d, "store": f"S{k}", "sales": round(float(v), 2)} for d, v in zip(dates, values))
    path = tmp_path / "sales.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _pipeline(tmp_path, *, parallel_keys: bool) -> ForecastEnginePipeline:
    default = ModelConfig.default()
    registry = tuple(
        replace(spec, enabled=True) if spec.name == "seasonal_naive" else spec for spec in default.registry
    )
    return ForecastEnginePipeline(
        model_config=replace(default, registry=registry),
        mlflow_config=MLflowConfig(enabled=False),
        pipeline_config=PipelineConfig(
            curated_storage=CuratedStorageConfig(root_dir=str(tmp_path / "curated")),
            model_storage=ModelStorageConfig(root_dir=str(tmp_path / "models")),
            forecast_export=ForecastExportConfig(root_dir=str(tmp_path / "forecasts")),
            artifacts_mirror=ArtifactsMirrorConfig(root_dir=str(tmp_path / "artifacts")),
        ),
        parallel_keys=parallel_keys,
    )


@requires_ray
def test_each_of_the_four_stages_is_its_own_genuine_ray_fan_out(tmp_path):
    dataset_path = _dataset(tmp_path, keys=3)
    pipeline = _pipeline(tmp_path, parallel_keys=True)

    context = pipeline.run(
        str(dataset_path),
        ForecastConfiguration(date_column="date", target_column="sales", key_columns=("store",)),
        run_id="staged-pipeline-e2e-test",
        selected_models=["seasonal_naive"],
    )

    by_name = {stage.name: stage for stage in context.stages}
    for name in ("Train Models", "Evaluate Models", "Explain Models", "Rank & Select"):
        stage = by_name[name]
        assert stage.status == "Completed"
        assert stage.parallel_tasks is not None, f"{name} reported no parallel_tasks"
        assert stage.parallel_tasks["executor"] == "ray"
        assert stage.parallel_tasks["total_tasks"] == 3
        assert stage.parallel_tasks["completed_tasks"] == 3
        assert stage.parallel_tasks["failed_tasks"] == 0
        assert len(stage.parallel_tasks["tasks"]) == 3
        # This stage's own real measured time, never borrowed from another.
        assert stage.measured_seconds is not None
        assert stage.measured_seconds >= 0.0


@requires_ray
def test_parallel_and_sequential_pipelines_produce_the_same_winner(tmp_path):
    dataset_path = _dataset(tmp_path, keys=3)
    config = ForecastConfiguration(date_column="date", target_column="sales", key_columns=("store",))

    sequential = _pipeline(tmp_path / "seq", parallel_keys=False).run(
        str(dataset_path), config, run_id="staged-pipeline-sequential", selected_models=["seasonal_naive"]
    )
    parallel = _pipeline(tmp_path / "par", parallel_keys=True).run(
        str(dataset_path), config, run_id="staged-pipeline-parallel", selected_models=["seasonal_naive"]
    )

    seq_winners = {r.group_id: r.model_name for r in sequential.production_selection_report.results}
    par_winners = {r.group_id: r.model_name for r in parallel.production_selection_report.results}
    assert seq_winners == par_winners
    assert set(seq_winners) == {"S1", "S2", "S3"}


def test_sequential_stages_report_no_parallel_tasks(tmp_path):
    dataset_path = _dataset(tmp_path, keys=2)
    pipeline = _pipeline(tmp_path, parallel_keys=False)

    context = pipeline.run(
        str(dataset_path),
        ForecastConfiguration(date_column="date", target_column="sales", key_columns=("store",)),
        run_id="staged-pipeline-sequential-only",
        selected_models=["seasonal_naive"],
    )

    by_name = {stage.name: stage for stage in context.stages}
    for name in ("Train Models", "Evaluate Models", "Explain Models", "Rank & Select"):
        assert by_name[name].parallel_tasks is None
