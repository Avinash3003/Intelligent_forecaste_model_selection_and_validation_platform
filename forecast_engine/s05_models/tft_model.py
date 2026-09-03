"""The deep-learning candidate, backed by pytorch-forecasting.

Its dependencies are multi-gigabyte and deliberately excluded from the
default install: a deployment that wants TFT installs pytorch-forecasting
explicitly, and one that does not sees TFT reported Unavailable while every
other model trains normally.
"""

from __future__ import annotations

import importlib.util
import tempfile
from typing import Any

import pandas as pd

from forecast_engine.s05_models.base_model import BaseForecastingModel, ForecastOutput
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries


_QUIET_TRAINER = {
    "logger": False,
    "enable_checkpointing": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "default_root_dir": tempfile.gettempdir(),
}


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
            accelerator="cpu",
            **_QUIET_TRAINER,
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

        raw = self._model.predict(dataloader, trainer_kwargs=dict(_QUIET_TRAINER))
        values = [float(value) for value in raw.flatten().tolist()]

        if len(values) >= horizon:
            return ForecastOutput(values=values[:horizon])

        # Hold the last predicted level if the decoder is shorter than the
        # requested horizon; the caller still receives a complete horizon.
        padding = [values[-1]] * (horizon - len(values)) if values else [0.0] * horizon
        return ForecastOutput(values=values + padding)

    # Strip what pickle cannot cross a process boundary with, right before
    # it tries.
    #
    # Every forward pass through the fitted network — training, and every
    # `predict()` call after it — leaves two things on `self._model` that
    # standard `pickle` refuses:
    #
    #   `_output_class`  a class pytorch-forecasting DEFINES INSIDE A
    #                     FUNCTION (`TupleOutputMixIn.to_network_output`)
    #                     and caches on the module for reuse. A class with
    #                     no importable module path cannot be pickled.
    #   `_trainer`        Lightning's own back-reference from the module to
    #                     the `Trainer` that fit it, which in turn chains
    #                     back through its loops and callbacks to the same
    #                     unpicklable class.
    #
    # Reproduced directly (no cluster needed): fit a TFT model, and
    # `pickle.dumps(model)` fails with exactly
    #   AttributeError: Can't pickle local object
    #   'TupleOutputMixIn.to_network_output.<locals>.Output'
    # every other model in this engine is a plain statistical or
    # scikit-learn-shaped object with none of this, which is why the bug
    # is specific to TFT.
    #
    # Deleting `_output_class` is safe — pytorch-forecasting's own
    # `to_network_output` gates on `hasattr(self, "_output_class")` and
    # rebuilds it (a cheap class definition, not a recomputation) the next
    # time it is needed, whether that is later in this same process or
    # after this object has been unpickled elsewhere. `_trainer` is set to
    # None for the same reason: pytorch-forecasting's own `.predict()`
    # does not require it. Verified empirically: predictions are
    # byte-identical before this cleanup, after it, and again after a full
    # pickle/unpickle round-trip.
    #
    # Mutates `self._model` in place rather than operating on a copy: both
    # attributes are cheap to regenerate, so leaving the live object
    # cleaned costs nothing and means a second `predict()` call followed
    # by a second pickle is handled the same way, automatically.
    def _strip_unpicklable_state(self) -> None:
        model = self.__dict__.get("_model")
        if model is not None:
            if hasattr(model, "_output_class"):
                del model._output_class
            model._trainer = None

    # Belt-and-suspenders: fires whenever THIS wrapper is pickled directly.
    # Not sufficient by itself — see prepare_for_pickling below for why.
    def __getstate__(self) -> dict[str, Any]:
        self._strip_unpicklable_state()
        return self.__dict__

    # `TrainedModel` (forecast_engine/s04_training/model_trainer.py) hands
    # the fitted network out through TWO separate top-level references —
    # `.model` (the bare `self._model`, for callers that want the raw
    # estimator) and `.fitted_model` (this wrapper, for callers that want
    # to call `.predict()` again) — and pickles both as plain dataclass
    # fields of the same object graph. `__getstate__` above only ever
    # fires for the SECOND one: pickle calls it because it is pickling
    # THIS wrapper object, but `.model` is a direct reference to the raw
    # pytorch-forecasting network, a different Python object whose own
    # class defines no `__getstate__` of its own — so pickling `.model`
    # bypasses this wrapper's cleanup entirely and fails on exactly the
    # same two attributes.
    #
    # The key-parallel executor (forecast_engine/parallel/ray_executor.py)
    # calls this on every trained model's wrapper, eagerly, right before
    # its one `pickle.dumps` call that ships a key's results back to the
    # Ray driver — cleaning `self._model` in place here fixes BOTH
    # references at once, since `.model` and `.fitted_model._model` are
    # the same live object, not copies.
    def prepare_for_pickling(self) -> None:
        self._strip_unpicklable_state()
