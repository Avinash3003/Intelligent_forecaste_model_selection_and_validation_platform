"""The deep-learning candidate, backed by pytorch-forecasting.

Its dependencies are multi-gigabyte and deliberately excluded from the
default install: a deployment that wants TFT installs pytorch-forecasting
explicitly, and one that does not sees TFT reported Unavailable while every
other model trains normally.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pandas as pd

from forecast_engine.s05_models.base_model import BaseForecastingModel, ForecastOutput
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries


class TemporalFusionTransformerModel(BaseForecastingModel):
    """Attention-based deep forecasting model via pytorch-forecasting."""

    # Whether pytorch-forecasting and torch are both importable
    @classmethod
    def is_available(cls) -> bool:
        return (
            importlib.util.find_spec("pytorch_forecasting") is not None
            and importlib.util.find_spec("torch") is not None
        )

    # Deferred to train(), which builds the network from the data itself
    def initialize(self) -> None:
        self._estimator = None

    # Build a TimeSeriesDataSet from the series and fit the TFT network
    def train(self, series: ForecastSeries) -> dict[str, Any]:
        from lightning.pytorch import Trainer
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        frame = series.frame.copy()

        # pytorch-forecasting indexes on a contiguous integer time step and
        # requires an explicit group id even for a single series.
        frame["time_idx"] = range(len(frame))
        frame["series_id"] = series.group_id
        target_column = series.target_column
        frame[target_column] = frame[target_column].astype(float)

        max_encoder_length = int(self.params.get("max_encoder_length", 24))
        max_prediction_length = int(self.params.get("max_prediction_length", 12))

        dataset = TimeSeriesDataSet(
            frame,
            time_idx="time_idx",
            target=target_column,
            group_ids=["series_id"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_unknown_reals=[target_column],
            time_varying_known_reals=self._usable_feature_columns(series),
            target_normalizer=GroupNormalizer(groups=["series_id"]),
            allow_missing_timesteps=True,
        )

        dataloader = dataset.to_dataloader(
            train=True, batch_size=int(self.params.get("batch_size", 64)), num_workers=0
        )

        network = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=float(self.params.get("learning_rate", 0.03)),
            hidden_size=int(self.params.get("hidden_size", 16)),
            attention_head_size=int(self.params.get("attention_head_size", 2)),
            dropout=float(self.params.get("dropout", 0.1)),
        )

        trainer = Trainer(
            max_epochs=int(self.params.get("max_epochs", 10)),
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
            accelerator="cpu",
        )
        trainer.fit(network, train_dataloaders=dataloader)

        self._model = network

        self._dataset = dataset
        self._training_frame = frame

        return {
            "observations": int(len(frame)),
            "max_encoder_length": max_encoder_length,
            "max_prediction_length": max_prediction_length,
            "epochs": int(self.params.get("max_epochs", 10)),
        }

    # Forecast horizon steps from the fitted network
    def predict(self, horizon: int, future_frame: pd.DataFrame | None = None) -> ForecastOutput:
        # pytorch-forecasting produces predictions for the decoder length the
        # dataset was built with, so the result is trimmed or padded here.
        self._require_trained()

        from pytorch_forecasting import TimeSeriesDataSet

        prediction_dataset = TimeSeriesDataSet.from_dataset(
            self._dataset, self._training_frame, predict=True, stop_randomization=True
        )
        dataloader = prediction_dataset.to_dataloader(
            train=False, batch_size=int(self.params.get("batch_size", 64)), num_workers=0
        )

        raw = self._model.predict(dataloader)
        values = [float(value) for value in raw.flatten().tolist()]

        if len(values) >= horizon:
            return ForecastOutput(values=values[:horizon])

        # Hold the last predicted level if the decoder is shorter than the
        # requested horizon; the caller still receives a complete horizon.
        padding = [values[-1]] * (horizon - len(values)) if values else [0.0] * horizon
        return ForecastOutput(values=values + padding)
