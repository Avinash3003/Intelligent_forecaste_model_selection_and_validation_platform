"""Priority C — user-selectable derived feature columns.

Covers the authoritative registry (`forecast_engine.config.
derived_features_config`) and that a resolved selection actually changes
what `SupervisedTreeModel` builds — not a cosmetic checkbox, a real
parameter that reaches `build_design_matrix()` and (separately)
`build_estimator()` never receives it as a stray keyword argument.
"""

from __future__ import annotations

import pandas as pd
import pytest

from forecast_engine.config.derived_features_config import (
    DEFAULT_SELECTED_FEATURE_IDS,
    SUPPORTED_FEATURE_IDS,
    apply_to_model_config,
    resolve_derived_feature_params,
    validate_feature_ids,
)
from forecast_engine.config.model_config import ModelConfig, ModelSpec
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.s05_models.xgboost_model import XGBoostModel


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------


def test_supported_feature_ids_match_the_model_adapters_own_defaults():
    # lag_1..lag_12, rolling_mean_3/6, month, quarter — exactly what
    # SupervisedTreeModel.DEFAULT_LAGS/DEFAULT_ROLLING_WINDOWS/calendar
    # already produce unprompted; nothing invented beyond that.
    assert SUPPORTED_FEATURE_IDS == {
        "lag_1", "lag_2", "lag_3", "lag_12",
        "rolling_mean_3", "rolling_mean_6",
        "month", "quarter",
    }


def test_validate_splits_valid_from_unsupported():
    valid, rejected = validate_feature_ids(["lag_1", "month", "lag_999", "not_a_feature"])
    assert valid == ["lag_1", "month"]
    assert rejected == ["lag_999", "not_a_feature"]


def test_no_selection_resolves_to_every_default_feature():
    # None means "this run never mentioned derived features at all" — every
    # run before this feature existed. Must reproduce current behavior
    # exactly: every supported feature on.
    params = resolve_derived_feature_params(None)
    assert set(params["lags"]) == {1, 2, 3, 12}
    assert set(params["rolling_windows"]) == {3, 6}
    assert set(params["calendar_features"]) == {"month", "quarter"}
    assert DEFAULT_SELECTED_FEATURE_IDS == SUPPORTED_FEATURE_IDS


def test_a_partial_selection_resolves_to_only_those_features():
    params = resolve_derived_feature_params(["lag_1", "month"])
    assert params["lags"] == [1]
    assert params["rolling_windows"] == []
    assert params["calendar_features"] == ["month"]


def test_an_empty_selection_resolves_to_nothing_not_the_default():
    params = resolve_derived_feature_params([])
    assert params == {"lags": [], "rolling_windows": [], "calendar_features": []}


def test_unsupported_ids_are_silently_dropped_at_the_engine_layer():
    # The API layer is where an unsupported name is rejected with an error
    # (never reaches here); the engine itself stays defensive regardless.
    params = resolve_derived_feature_params(["lag_1", "not_a_real_feature"])
    assert params["lags"] == [1]


# ---------------------------------------------------------------------
# SupervisedTreeModel actually honoring a resolved selection
# ---------------------------------------------------------------------


def _series(rows: int = 30) -> ForecastSeries:
    dates = pd.date_range("2021-01-01", periods=rows, freq="MS")
    frame = pd.DataFrame({"date": dates, "sales": [100.0 + i for i in range(rows)]})
    return ForecastSeries(group_id="g1", frame=frame, date_column="date", target_column="sales")


def _spec(params: dict) -> ModelSpec:
    return ModelSpec(
        name="xgboost",
        adapter="forecast_engine.s05_models.xgboost_model.XGBoostModel",
        default_params={"n_estimators": 10, "max_depth": 2, "random_state": 42, "n_jobs": 1, **params},
    )


def test_a_full_selection_produces_every_expected_column():
    params = resolve_derived_feature_params(["lag_1", "lag_2", "rolling_mean_3", "month", "quarter"])
    model = XGBoostModel(_spec(params))
    model.initialize()
    features, _target = model.build_design_matrix(_series())

    for expected in ("lag_1", "lag_2", "rolling_mean_3", "month", "quarter"):
        assert expected in features.columns
    assert "lag_3" not in features.columns
    assert "lag_12" not in features.columns
    assert "rolling_mean_6" not in features.columns


def test_deselecting_every_derived_feature_leaves_none_of_them():
    params = resolve_derived_feature_params([])
    model = XGBoostModel(_spec(params))
    model.initialize()
    features, _target = model.build_design_matrix(_series())

    for absent in ("lag_1", "lag_2", "lag_3", "lag_12", "rolling_mean_3", "rolling_mean_6", "month", "quarter"):
        assert absent not in features.columns
    # The mandatory ordinal feature every tree model needs regardless.
    assert "time_index" in features.columns


def test_the_default_selection_matches_pre_priority_c_behavior():
    default_model = XGBoostModel(_spec({}))  # no params at all — the old, pre-feature-selection path
    default_model.initialize()
    default_features, _ = default_model.build_design_matrix(_series())

    resolved_params = resolve_derived_feature_params(None)
    resolved_model = XGBoostModel(_spec(resolved_params))
    resolved_model.initialize()
    resolved_features, _ = resolved_model.build_design_matrix(_series())

    assert set(default_features.columns) == set(resolved_features.columns)


def test_feature_engineering_keys_never_reach_the_underlying_estimator():
    # These would raise "unexpected keyword argument" from XGBRegressor's
    # own constructor if build_estimator() passed self.params verbatim.
    params = resolve_derived_feature_params(["lag_1"])
    model = XGBoostModel(_spec(params))
    model.initialize()
    estimator = model.build_estimator()  # must not raise
    assert estimator is not None


@pytest.mark.parametrize("bad_selection", [["lag_1", "not_a_feature"]])
def test_an_unsupported_id_never_silently_becomes_a_real_feature(bad_selection):
    valid, rejected = validate_feature_ids(bad_selection)
    assert rejected == ["not_a_feature"]
    assert "not_a_feature" not in SUPPORTED_FEATURE_IDS


# ---------------------------------------------------------------------
# apply_to_model_config — the ModelConfig-level wiring run_pipeline.py uses
# ---------------------------------------------------------------------


def test_none_selection_returns_the_identical_model_config_object():
    config = ModelConfig.default()
    assert apply_to_model_config(config, None) is config


def test_selection_only_touches_xgboost_and_lightgbm_specs():
    config = ModelConfig.default()
    updated = apply_to_model_config(config, ["lag_1"])

    before_by_name = {spec.name: spec for spec in config.registry}
    after_by_name = {spec.name: spec for spec in updated.registry}

    for name in ("prophet", "arima", "tft", "seasonal_naive"):
        assert after_by_name[name].default_params == before_by_name[name].default_params

    assert after_by_name["xgboost"].default_params["lags"] == [1]
    assert after_by_name["lightgbm"].default_params["lags"] == [1]
    # Every other xgboost/lightgbm hyperparameter (n_estimators, etc.)
    # survives the merge untouched.
    assert after_by_name["xgboost"].default_params["n_estimators"] == before_by_name["xgboost"].default_params["n_estimators"]


def test_apply_to_model_config_result_is_what_the_trainer_would_actually_use():
    config = ModelConfig.default()
    updated = apply_to_model_config(config, ["lag_1", "lag_2"])
    spec = next(spec for spec in updated.registry if spec.name == "xgboost")

    model = XGBoostModel(spec)
    model.initialize()
    dates = pd.date_range("2021-01-01", periods=30, freq="MS")
    frame = pd.DataFrame({"date": dates, "sales": [float(i) for i in range(30)]})
    series = ForecastSeries(group_id="g1", frame=frame, date_column="date", target_column="sales")
    features, _ = model.build_design_matrix(series)

    assert "lag_1" in features.columns
    assert "lag_2" in features.columns
    assert "lag_3" not in features.columns


# ---------------------------------------------------------------------
# End-to-end: a real pipeline run with a derived-feature selection
# ---------------------------------------------------------------------


def test_a_real_run_trains_successfully_with_a_derived_feature_selection(tmp_path):
    from forecast_engine.config.mlflow_config import MLflowConfig
    from forecast_engine.config.pipeline_config import (
        ArtifactsMirrorConfig,
        CuratedStorageConfig,
        ForecastExportConfig,
        ModelStorageConfig,
        PipelineConfig,
    )
    from forecast_engine.core.forecast_configuration import ForecastConfiguration
    from forecast_engine.run_pipeline import ForecastEnginePipeline

    dates = pd.date_range("2021-01-01", periods=30, freq="MS")
    frame = pd.DataFrame({"date": dates, "sales": [100.0 + i * 2 for i in range(30)]})
    dataset_path = tmp_path / "sales.csv"
    frame.to_csv(dataset_path, index=False)

    model_config = apply_to_model_config(ModelConfig.default(), ["lag_1", "month"])
    pipeline = ForecastEnginePipeline(
        model_config=model_config,
        mlflow_config=MLflowConfig(enabled=False),
        pipeline_config=PipelineConfig(
            curated_storage=CuratedStorageConfig(root_dir=str(tmp_path / "curated")),
            model_storage=ModelStorageConfig(root_dir=str(tmp_path / "models")),
            forecast_export=ForecastExportConfig(root_dir=str(tmp_path / "forecasts")),
            artifacts_mirror=ArtifactsMirrorConfig(root_dir=str(tmp_path / "artifacts")),
        ),
    )

    context = pipeline.run(
        str(dataset_path),
        ForecastConfiguration(date_column="date", target_column="sales"),
        run_id="derived-features-e2e-test",
        selected_models=["xgboost"],
        derived_features=["lag_1", "month"],
    )

    assert context.derived_features == ["lag_1", "month"]
    assert context.summary()["derived_features"] == ["lag_1", "month"]
    trained = context.training_report.trained_models()
    assert trained and trained[0].status.value == "Trained"
