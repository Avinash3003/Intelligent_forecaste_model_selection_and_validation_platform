"""A trained TFT model must survive pickling, or it can never leave the
Ray worker that trained it.

Reproduced directly against the real dependencies — no cluster needed:

    fit a TFT model, then pickle.dumps(model)
    -> AttributeError: Can't pickle local object
       'TupleOutputMixIn.to_network_output.<locals>.Output'

Confirmed on a real DCS run too (dbx-run-bbfa4f809de0): the key task
raised this exact error inside `ray_executor._remote_run_key`'s
`pickle.dumps(reports, ...)`, so TFT trained but never made it back to
the driver -- 0 keys succeeded, 0 models registered.

Root cause: every forward pass through the fitted network leaves two
things on it that plain pickle refuses --

    `_output_class`  a class pytorch-forecasting defines INSIDE A
                      FUNCTION (`TupleOutputMixIn.to_network_output`) and
                      caches on the module. A class with no importable
                      module path cannot be pickled.
    `_trainer`        Lightning's own back-reference to the `Trainer`
                      that fit it, chaining back through its loops and
                      callbacks to the same unpicklable class.

No other model in this engine has either problem: everything else is a
plain statistical or scikit-learn-shaped object.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_forecasting")

from forecast_engine.config.model_config import ModelConfig
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.s05_models.tft_model import TemporalFusionTransformerModel


@pytest.fixture(scope="module")
def fitted_model():
    """Fit once per test module -- an epoch of TFT training is not free,
    and every test here needs the same trained object."""
    spec = ModelConfig.default().find("tft")
    n = 40
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    rng = np.random.RandomState(7)
    frame = pd.DataFrame(
        {"date": dates, "sales": 80 + 10 * np.sin(np.arange(n) / 3) + rng.rand(n)}
    )
    series = ForecastSeries(
        group_id="S1", frame=frame, date_column="date", target_column="sales", feature_columns=[]
    )
    model = TemporalFusionTransformerModel(
        spec, params={"max_epochs": 1, "max_encoder_length": 12, "max_prediction_length": 6}
    )
    model.initialize()
    model.train(series)
    return model


def test_a_freshly_trained_model_pickles(fitted_model):
    """Before any prediction, the state pickle would ship is already the
    one that has to survive Ray -- must not depend on predict() having
    been called first."""
    pickle.dumps(fitted_model, protocol=pickle.HIGHEST_PROTOCOL)


def test_a_model_that_has_predicted_still_pickles(fitted_model):
    """Every forward pass re-creates the unpicklable class, and evaluation
    calls predict() at least once (a backtest window) before the model is
    shipped back to the driver."""
    fitted_model.predict(horizon=6)

    pickle.dumps(fitted_model, protocol=pickle.HIGHEST_PROTOCOL)


def test_predictions_are_unaffected_by_the_pickle_safety_cleanup(fitted_model):
    """The fix mutates the live network object -- it must not change what
    the model actually predicts."""
    before = fitted_model.predict(horizon=6).values

    pickle.dumps(fitted_model, protocol=pickle.HIGHEST_PROTOCOL)
    after = fitted_model.predict(horizon=6).values

    assert before == after


def test_a_restored_model_predicts_identically_to_the_original(fitted_model):
    """The exact round-trip Ray performs: dumps on the worker, loads on
    the driver, and the model must still forecast the same numbers."""
    fitted_model.predict(horizon=6)
    expected = fitted_model.predict(horizon=6).values

    restored = pickle.loads(pickle.dumps(fitted_model, protocol=pickle.HIGHEST_PROTOCOL))

    assert restored.predict(horizon=6).values == expected


def test_the_model_survives_a_second_pickle_after_being_used_again(fitted_model):
    """model_writer.py pickles the winning model a second time, driver-
    side, to persist it. Nothing may assume the cleanup only ever runs
    once."""
    fitted_model.predict(horizon=6)
    once = pickle.loads(pickle.dumps(fitted_model, protocol=pickle.HIGHEST_PROTOCOL))

    once.predict(horizon=6)  # re-dirties _output_class on the restored object
    twice = pickle.loads(pickle.dumps(once, protocol=pickle.HIGHEST_PROTOCOL))

    assert twice.predict(horizon=6).values == once.predict(horizon=6).values


def test_a_key_shaped_container_survives_pickling_like_ray_ships_it():
    """Mirrors ray_executor._remote_run_key exactly: the fitted model
    travels inside a larger results object, not pickled on its own."""
    spec = ModelConfig.default().find("tft")
    n = 40
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    rng = np.random.RandomState(3)
    frame = pd.DataFrame(
        {"date": dates, "sales": 80 + 10 * np.sin(np.arange(n) / 3) + rng.rand(n)}
    )
    series = ForecastSeries(
        group_id="S2", frame=frame, date_column="date", target_column="sales", feature_columns=[]
    )
    model = TemporalFusionTransformerModel(
        spec, params={"max_epochs": 1, "max_encoder_length": 12, "max_prediction_length": 6}
    )
    model.initialize()
    model.train(series)
    forecast = model.predict(horizon=6)

    payload = pickle.dumps(
        {"fitted_model": model, "forecast": forecast}, protocol=pickle.HIGHEST_PROTOCOL
    )
    restored = pickle.loads(payload)

    assert restored["fitted_model"].predict(horizon=6).values == restored["forecast"].values


# ---------------------------------------------------------------------
# The gap the tests above all missed: TrainedModel.model is a SEPARATE,
# direct reference to the raw fitted estimator (model_trainer.py's
# `record.model = model.model`), pickled as its own top-level dataclass
# field alongside `record.fitted_model` (this wrapper). `__getstate__`
# above only ever fires for the wrapper being pickled — a bare reference
# to the raw pytorch-forecasting network has no `__getstate__` of its own,
# so every test above passed on real DCS while the actual production path
# (TrainedModel -> KeyReports -> ray_executor._remote_run_key) still
# raised the exact original AttributeError. Confirmed live on
# dbx-run-a7e7da5a87d2 after the __getstate__-only fix was already
# deployed. These reproduce that specific gap with the real dataclasses,
# not a hand-rolled container, and pin the real fix: an eager
# `prepare_for_pickling()` call the executor makes right before its one
# `pickle.dumps`, which cleans the shared object both references point to.
# ---------------------------------------------------------------------


def _fast_tft_spec():
    """The real 'tft' spec, with a short encoder/prediction window and one
    epoch -- everything else about it (adapter, defaults) stays real."""
    spec = ModelConfig.default().find("tft")
    return spec.__class__(
        **{
            **spec.__dict__,
            "min_observations": 1,
            "default_params": {
                **spec.default_params,
                "max_epochs": 1,
                "max_encoder_length": 12,
                "max_prediction_length": 6,
            },
        }
    )


def _fit_tft_trained_model_record():
    from forecast_engine.s04_training.model_trainer import ModelTrainer

    n = 40
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    rng = np.random.RandomState(11)
    frame = pd.DataFrame(
        {"date": dates, "sales": 80 + 10 * np.sin(np.arange(n) / 3) + rng.rand(n)}
    )
    series = ForecastSeries(
        group_id="S3", frame=frame, date_column="date", target_column="sales", feature_columns=[]
    )
    fast_spec = _fast_tft_spec()
    model_config = ModelConfig(registry=(fast_spec,))
    trainer = ModelTrainer(model_config)
    return trainer._train_one(fast_spec, series, available=True)


def test_the_raw_model_reference_alone_reproduces_the_real_bug_without_the_hook():
    """`.model` is what actually failed on real DCS -- this proves it fails
    on its own, with no fitted_model in the payload to (incorrectly) get
    credit for protecting it."""
    record = _fit_tft_trained_model_record()
    assert record.is_trained
    record.fitted_model.predict(horizon=6)  # the eval/forward-forecast pass that re-dirties it

    with pytest.raises((AttributeError, TypeError)):
        pickle.dumps({"model": record.model}, protocol=pickle.HIGHEST_PROTOCOL)


def test_trained_model_survives_pickling_once_prepare_for_pickling_runs():
    """The real fix: call the wrapper's hook once, eagerly, before
    pickling -- exactly what ray_executor._prepare_trained_models_for_pickling
    does -- and both of TrainedModel's references to the estimator survive."""
    record = _fit_tft_trained_model_record()
    record.fitted_model.predict(horizon=6)

    record.fitted_model.prepare_for_pickling()
    payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(payload)

    assert restored.model is not None
    assert restored.fitted_model.predict(horizon=6).values == record.fitted_model.predict(horizon=6).values


def test_ray_executors_own_prepare_for_pickling_hook_fixes_a_real_training_report():
    """End-to-end through the actual production code, not a
    re-implementation of it: forecast_engine.parallel.key_workflow.train_key()
    (the same call ray_executor._remote_train makes) produces a real
    TrainingReport carrying a real TrainedModel with both references
    populated, and ray_executor._prepare_trained_models_for_pickling is the
    exact function the executor calls right before that task returns."""
    from forecast_engine.config.drift_config import DriftValidationConfig
    from forecast_engine.config.evaluation_config import EvaluationConfig
    from forecast_engine.config.explainability_config import ExplainabilityConfig
    from forecast_engine.config.ranking_config import RankingConfig
    from forecast_engine.parallel.key_workflow import KeyWorkflowConfig, evaluate_key, train_key
    from forecast_engine.parallel.ray_executor import _prepare_trained_models_for_pickling

    n = 40
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    rng = np.random.RandomState(5)
    frame = pd.DataFrame(
        {"date": dates, "sales": 80 + 10 * np.sin(np.arange(n) / 3) + rng.rand(n)}
    )
    series = ForecastSeries(
        group_id="S4", frame=frame, date_column="date", target_column="sales", feature_columns=[]
    )
    model_config = ModelConfig(registry=(_fast_tft_spec(),))
    config = KeyWorkflowConfig(
        model=model_config,
        evaluation=EvaluationConfig(),
        explainability=ExplainabilityConfig(),
        ranking=RankingConfig(),
        drift=DriftValidationConfig(),
        selected_models=("tft",),
    )

    training = train_key(series, config)
    trained = [r for r in training.results if r.is_trained]
    assert trained, "the group must have actually trained for this test to mean anything"

    _prepare_trained_models_for_pickling(training)
    # The exact boundary the real bug crossed: pickled as a Ray task's
    # RETURN value here, then unpickled and handed to the NEXT stage's
    # task as an ARGUMENT -- two separate crossings, one cleanup.
    restored_training = pickle.loads(pickle.dumps(training, protocol=pickle.HIGHEST_PROTOCOL))
    restored_trained = restored_training.trained_models()

    evaluation = evaluate_key(series, config, restored_trained)
    pickle.dumps(evaluation, protocol=pickle.HIGHEST_PROTOCOL)  # must not raise
