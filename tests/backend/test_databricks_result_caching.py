"""Section 6.14 performance review: a completed Databricks run's
`summary.json` must be downloaded at most once per process, no matter how
many times its Results page is opened, its business key is switched, or
its debug/LLMOps panels are read.

`summary.json` grows with a run's group/model count — for a large run this
is a genuinely large remote artifact download. `DatabricksRunner.get_result()`
already caches it on `_DatabricksJobRecord.summary`; these tests exist so a
future change to that path fails loudly here instead of silently
reintroducing a per-request download.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.orchestration.databricks_runner import DatabricksRunner
from app.orchestration.schemas import JobStatus, PipelineExecutionRequest


class _CountingFiles:
    """Like the shared `_FakeFiles` fake, but counts every download by path
    so a test can assert a given file was fetched exactly once."""

    def __init__(self) -> None:
        self.uploaded: dict[str, bytes] = {}
        self.download_calls: list[str] = []

    def upload(self, file_path, contents, overwrite=None):
        self.uploaded[file_path] = contents.read()

    def download(self, file_path):
        self.download_calls.append(file_path)
        if file_path not in self.uploaded:
            raise FileNotFoundError(file_path)
        return SimpleNamespace(contents=SimpleNamespace(read=lambda: self.uploaded[file_path]))


class _CountingJobs:
    def __init__(self) -> None:
        self.get_run_calls = 0
        self._state = SimpleNamespace(life_cycle_state="TERMINATED", result_state="SUCCESS", state_message="")

    def list(self, name=None):
        return [SimpleNamespace(job_id=4242, settings=SimpleNamespace(name=name))]

    def run_now(self, job_id, job_parameters=None):
        return SimpleNamespace(run_id=99)

    def get_run(self, run_id, **kwargs):
        self.get_run_calls += 1
        return SimpleNamespace(state=self._state, run_duration=12_000)

    def cancel_run(self, run_id):
        pass


class _CountingWorkspace:
    def __init__(self) -> None:
        self.files = _CountingFiles()
        self.jobs = _CountingJobs()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        execution_mode="databricks",
        databricks_host="https://example.invalid",
        databricks_token="test-token",
        databricks_volumes_root="/Volumes/forecastiq/forecasting/forecast_files",
        upload_dir=str(tmp_path / "uploads"),
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("date,store,sales\n2024-01-01,1,10\n")
    return path


def _request(dataset):
    return PipelineExecutionRequest(
        run_id="dbx-run-cache-test",
        dataset_path=str(dataset),
        dataset_name="sales.csv",
        forecast_configuration={
            "date_column": "date",
            "target_column": "sales",
            "key_columns": ["store"],
            "feature_columns": [],
            "aggregation_method": "sum",
        },
        selected_models=["prophet"],
        horizon=12,
    )


def _complete_run(runner: DatabricksRunner, workspace: _CountingWorkspace, dataset) -> str:
    run_id = runner.submit(_request(dataset))
    root = f"/Volumes/forecastiq/forecasting/forecast_files/runs/{run_id}"
    workspace.files.uploaded[f"{root}/summary.json"] = json.dumps(
        {
            "run_id": run_id,
            "production_selection_report": {"results": []},
            "forecast_groups": [],
        }
    ).encode()
    return run_id


def test_summary_json_is_downloaded_at_most_once_across_repeated_get_result_calls(settings, dataset):
    workspace = _CountingWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = _complete_run(runner, workspace, dataset)

    # Simulates the Results page's initial load, then a user switching
    # business key twice — each is a separate GET /results/{run_id} call.
    for _ in range(3):
        result = runner.get_result(run_id)
        assert result.job_status is JobStatus.COMPLETED

    summary_downloads = [p for p in workspace.files.download_calls if p.endswith("summary.json")]
    assert len(summary_downloads) == 1


def test_status_polling_after_completion_reads_neither_jobs_api_nor_the_volume(settings, dataset):
    """`_refresh()` must be a no-op once a run is terminal — the frontend's
    3-second poll must not keep calling `jobs.get_run()` or re-reading
    `live_status.json` for a run that already finished."""
    workspace = _CountingWorkspace()
    runner = DatabricksRunner(settings, workspace_client=workspace)
    run_id = _complete_run(runner, workspace, dataset)

    runner.get_status(run_id)  # first call after TERMINATED transitions the record
    calls_after_first = workspace.jobs.get_run_calls
    downloads_after_first = len(workspace.files.download_calls)

    for _ in range(5):
        assert runner.get_status(run_id) is JobStatus.COMPLETED

    assert workspace.jobs.get_run_calls == calls_after_first
    assert len(workspace.files.download_calls) == downloads_after_first


def test_workspace_client_is_built_with_conservative_explicit_timeouts(settings, monkeypatch):
    """The SDK's own defaults (300s retry budget) are tuned for a one-off
    CLI call, not a 3-second status-poll cadence — see `_workspace`'s
    docstring/constants. This checks the values `DatabricksRunner` passes
    to the SDK's `Config`, without constructing a real `WorkspaceClient`
    (which would attempt real host-metadata network calls for a fake host).
    """
    import databricks.sdk as sdk_module

    captured: dict = {}

    class _RecordingConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _NoopWorkspaceClient:
        def __init__(self, config=None):
            self.config = config

    monkeypatch.setattr("databricks.sdk.config.Config", _RecordingConfig)
    monkeypatch.setattr(sdk_module, "WorkspaceClient", _NoopWorkspaceClient)

    runner = DatabricksRunner(settings)
    client = runner._workspace  # noqa: SLF001 - the property under test

    assert captured["http_timeout_seconds"] == DatabricksRunner._WORKSPACE_HTTP_TIMEOUT_SECONDS
    assert captured["retry_timeout_seconds"] == DatabricksRunner._WORKSPACE_RETRY_TIMEOUT_SECONDS
    assert isinstance(client.config, _RecordingConfig)
