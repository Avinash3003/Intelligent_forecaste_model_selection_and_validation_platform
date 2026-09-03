"""The job installs the wheel this deployment published, not a fixed name.

CI stamps every build with its own version (forecast_engine/_version.py), so
the deployed filename changes each deploy while the setting naming it does
not. That mismatch terminated a job cluster with ERROR_NO_SUCH_FILE_OR_
DIRECTORY: the bundle had published forecast_engine-0.1.0+ci.53 while the
job still asked for forecast_engine-0.1.0.

Resolving from the directory keeps the two in step without a second
packaging mechanism, and without CI having to write a wheel filename back
into application configuration.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.orchestration.databricks_runner import DatabricksRunner

INTERNAL = "/Workspace/Users/sp/.bundle/forecastiq/prod/artifacts/.internal"


class _Workspace:
    def __init__(self, paths, error=None):
        self._paths = list(paths)
        self._error = error
        self.listed: list[str] = []

    @property
    def workspace(self):
        outer = self

        class _Api:
            def list(self, directory):
                outer.listed.append(directory)
                if outer._error:
                    raise outer._error
                return [SimpleNamespace(path=p) for p in outer._paths]

        return _Api()


def _runner(configured, workspace):
    settings = Settings(
        execution_mode="databricks",
        databricks_host="https://example.invalid",
        databricks_token="t",
        databricks_engine_wheel_path=configured,
    )
    return DatabricksRunner(settings, workspace_client=workspace)


def test_a_stale_filename_resolves_to_what_is_deployed_now():
    """The reported failure: the job asked for a wheel a later deploy replaced."""
    workspace = _Workspace([f"{INTERNAL}/forecast_engine-0.1.0+ci.53-py3-none-any.whl"])
    runner = _runner(f"{INTERNAL}/forecast_engine-0.1.0-py3-none-any.whl", workspace)

    assert runner._resolve_engine_wheel() == f"{INTERNAL}/forecast_engine-0.1.0+ci.53-py3-none-any.whl"
    assert workspace.listed == [INTERNAL]


def test_a_directory_resolves_to_the_wheel_inside_it():
    """The setting can name the bundle's artifact directory outright."""
    workspace = _Workspace([f"{INTERNAL}/forecast_engine-0.1.0+ci.53-py3-none-any.whl"])

    assert _runner(INTERNAL, workspace)._resolve_engine_wheel().endswith("ci.53-py3-none-any.whl")


def test_a_configured_wheel_that_still_exists_is_used_as_given():
    exact = f"{INTERNAL}/forecast_engine-0.1.0+ci.53-py3-none-any.whl"
    workspace = _Workspace([exact, f"{INTERNAL}/forecast_engine-0.1.0-py3-none-any.whl"])

    assert _runner(exact, workspace)._resolve_engine_wheel() == exact


def test_the_newest_ci_build_wins_when_several_remain():
    """By version, not by name: '+' sorts before '-', so plain 0.1.0 would
    otherwise beat every CI build and reinstate the stale-wheel problem."""
    workspace = _Workspace(
        [
            f"{INTERNAL}/forecast_engine-0.1.0-py3-none-any.whl",
            f"{INTERNAL}/forecast_engine-0.1.0+ci.7-py3-none-any.whl",
            f"{INTERNAL}/forecast_engine-0.1.0+ci.53-py3-none-any.whl",
        ]
    )

    resolved = _runner(INTERNAL, workspace)._resolve_engine_wheel()

    assert resolved.endswith("forecast_engine-0.1.0+ci.53-py3-none-any.whl")


def test_nothing_configured_installs_nothing():
    """Existing Compute may already carry the engine; that is not an error."""
    workspace = _Workspace([])
    runner = _runner(None, workspace)

    assert runner._resolve_engine_wheel() == ""
    assert workspace.listed == []


def test_an_unlistable_directory_falls_back_to_the_configured_path():
    """A listing failure must not strip the library from the job."""
    exact = f"{INTERNAL}/forecast_engine-0.1.0+ci.53-py3-none-any.whl"
    workspace = _Workspace([], error=RuntimeError("PERMISSION_DENIED"))

    assert _runner(exact, workspace)._resolve_engine_wheel() == exact


def test_an_empty_directory_falls_back_rather_than_dropping_the_library():
    exact = f"{INTERNAL}/forecast_engine-0.1.0+ci.53-py3-none-any.whl"
    workspace = _Workspace([f"{INTERNAL}/notes.txt"])

    assert _runner(exact, workspace)._resolve_engine_wheel() == exact


def test_the_resolved_wheel_is_what_the_task_installs():
    compute_sdk = SimpleNamespace(Library=lambda whl: SimpleNamespace(whl=whl))
    workspace = _Workspace([f"{INTERNAL}/forecast_engine-0.1.0+ci.53-py3-none-any.whl"])
    runner = _runner(f"{INTERNAL}/forecast_engine-0.1.0-py3-none-any.whl", workspace)

    libraries = runner._engine_libraries(compute_sdk)

    assert [lib.whl for lib in libraries] == [f"{INTERNAL}/forecast_engine-0.1.0+ci.53-py3-none-any.whl"]
