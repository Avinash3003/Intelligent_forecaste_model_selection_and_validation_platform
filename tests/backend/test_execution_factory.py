"""build_runner()'s mode dispatch — every execution_mode resolves to the
Runner it names, explicitly, with no silent fallback to another backend.
"""

import pytest

from app.config.settings import Settings
from app.orchestration.dcs_runner import DcsRunner
from app.orchestration.databricks_runner import DatabricksRunner
from app.orchestration.exceptions import RunnerConfigurationError
from app.orchestration.executor import build_runner
from app.orchestration.local_runner import LocalRunner


def _settings(tmp_path, **overrides):
    return Settings(
        upload_dir=str(tmp_path / "uploads"),
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        **overrides,
    )


def test_local_mode_builds_local_runner(tmp_path):
    runner = build_runner(_settings(tmp_path, execution_mode="local"))
    assert isinstance(runner, LocalRunner)


def test_databricks_mode_builds_databricks_runner(tmp_path):
    runner = build_runner(
        _settings(tmp_path, execution_mode="databricks", databricks_host="https://example.invalid", databricks_token="t")
    )
    assert isinstance(runner, DatabricksRunner)
    assert not isinstance(runner, DcsRunner)


def test_databricks_dcs_mode_builds_dcs_runner_not_local_runner(tmp_path):
    """The regression this test exists for: an `else: LocalRunner()`
    fallthrough would silently run a cloud mode locally instead of failing.
    """
    runner = build_runner(
        _settings(
            tmp_path,
            execution_mode="databricks_dcs",
            databricks_host="https://example.invalid",
            databricks_token="t",
        )
    )
    assert isinstance(runner, DcsRunner)
    assert not isinstance(runner, LocalRunner)


def test_unknown_execution_mode_fails_clearly_instead_of_defaulting(tmp_path):
    with pytest.raises(RunnerConfigurationError):
        build_runner(_settings(tmp_path, execution_mode="staging"))


def test_dcs_runner_targets_the_dcs_job_by_default_name(tmp_path):
    runner = build_runner(
        _settings(
            tmp_path,
            execution_mode="databricks_dcs",
            databricks_host="https://example.invalid",
            databricks_token="t",
        )
    )
    assert runner._job_name == "forecastiq-forecast-pipeline-dcs"


def test_serverless_and_dcs_runners_target_different_job_names(tmp_path):
    serverless = build_runner(
        _settings(tmp_path, execution_mode="databricks", databricks_host="https://example.invalid", databricks_token="t")
    )
    dcs = build_runner(
        _settings(
            tmp_path,
            execution_mode="databricks_dcs",
            databricks_host="https://example.invalid",
            databricks_token="t",
        )
    )
    assert serverless._job_name != dcs._job_name
