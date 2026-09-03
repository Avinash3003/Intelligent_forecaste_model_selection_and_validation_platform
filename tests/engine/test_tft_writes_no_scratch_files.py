"""TFT must not write Lightning scratch into the working directory.

pytorch_forecasting's `BaseModel.predict` builds its own Trainer with
Lightning's default logger, so every prediction wrote another
lightning_logs/version_N holding one hparams.yaml. A 500-key run left
hundreds of them in the repo, and in a Databricks task's working directory.

Silencing only the training Trainer was not enough; the prediction one is a
separate Trainer this code does not own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_engine.config.model_config import ModelConfig
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.s05_models.tft_model import _QUIET_TRAINER, TemporalFusionTransformerModel


def _series() -> ForecastSeries:
    dates = pd.date_range("2020-01-01", periods=40, freq="MS")
    values = 80 + 10 * np.sin(np.arange(40) / 3) + np.random.RandomState(7).rand(40)
    return ForecastSeries(
        group_id="S1",
        frame=pd.DataFrame({"date": dates, "sales": values}),
        date_column="date",
        target_column="sales",
        feature_columns=[],
    )


def test_training_and_predicting_leave_the_working_directory_clean(tmp_path, monkeypatch):
    pytest.importorskip("pytorch_forecasting")
    monkeypatch.chdir(tmp_path)

    model = TemporalFusionTransformerModel(
        ModelConfig.default().find("tft"),
        params={"max_epochs": 1, "max_encoder_length": 12, "max_prediction_length": 6},
    )
    model.initialize()
    model.train(_series())
    model.predict(6)

    assert not (tmp_path / "lightning_logs").exists()
    assert list(tmp_path.iterdir()) == []


def test_the_prediction_trainer_is_silenced_too():
    import inspect

    source = inspect.getsource(TemporalFusionTransformerModel.predict)

    assert "trainer_kwargs" in source
    assert _QUIET_TRAINER["logger"] is False
    assert _QUIET_TRAINER["enable_checkpointing"] is False
