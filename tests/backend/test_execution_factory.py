"""build_runner()'s mode dispatch — every execution_mode resolves to the
Runner it names, explicitly, with no silent fallback to another backend.
"""

import pytest

from app.config.settings import Settings
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
    assert not isinstance(runner, LocalRunner)



def test_unknown_execution_mode_fails_clearly_instead_of_defaulting(tmp_path):
    with pytest.raises(RunnerConfigurationError):
        build_runner(_settings(tmp_path, execution_mode="staging"))



