"""The engine's `--config` file may carry run-level options.

This is what lets the Databricks job pass every per-run value through one
file instead of through fixed job parameters that cannot be left out.
"""

import json

import pytest

from forecast_engine.run_pipeline import _parse_args, apply_config_run_options, load_config_payload


def _args(config_path, extra=()):
    return _parse_args(["--dataset", "d.csv", "--config", str(config_path), *extra])


def _write(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


BASE = {"date_column": "date", "target_column": "sales", "key_columns": ["store", "item"]}


def test_run_options_are_read_from_the_config_file(tmp_path):
    path = _write(tmp_path, {
        **BASE,
        "run_id": "dbx-run-1",
        "dataset_name": "sales.csv",
        "models": ["prophet", "xgboost"],
        "fallback_model": "xgboost",
        "horizon": 24,
    })
    args = _args(path)
    apply_config_run_options(args, load_config_payload(args))

    assert args.run_id == "dbx-run-1"
    assert args.dataset_name == "sales.csv"
    assert args.models == ["prophet", "xgboost"]
    assert args.fallback_model == "xgboost"
    assert args.horizon == 24


def test_an_explicit_flag_always_beats_the_config_file(tmp_path):
    # Local invocations must behave exactly as they did before this existed.
    path = _write(tmp_path, {**BASE, "horizon": 24, "models": ["prophet"]})
    args = _args(path, ["--horizon", "36", "--models", "arima"])
    apply_config_run_options(args, load_config_payload(args))

    assert args.horizon == 36
    assert args.models == ["arima"]


def test_a_config_without_run_options_changes_nothing(tmp_path):
    args = _args(_write(tmp_path, BASE))
    apply_config_run_options(args, load_config_payload(args))

    assert args.run_id is None
    assert not args.models
    assert args.horizon is None


def test_an_out_of_range_horizon_is_rejected(tmp_path):
    from forecast_engine.utils.exceptions import ConfigurationError

    args = _args(_write(tmp_path, {**BASE, "horizon": 500}))
    with pytest.raises(ConfigurationError):
        apply_config_run_options(args, load_config_payload(args))


def test_multi_valued_key_columns_survive_the_config_file(tmp_path):
    from forecast_engine.run_pipeline import build_configuration_from_args

    configuration = build_configuration_from_args(_args(_write(tmp_path, BASE)))
    # The reason a config file is used at all: one flat job parameter
    # cannot express a composite business key.
    assert configuration.key_columns == ("store", "item")
