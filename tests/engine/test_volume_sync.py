"""A container run must land its outputs in the storage account.

The defect this pins: a Databricks Container Services image has no UC
Volumes mount (`Unrecognized storage scheme: uc-volumes`, and every path
under /Volumes answers `PermissionError [Errno 1]`), so container runs
write to workspace files instead. That took every output out of the UC
Volume the storage account sits behind — run summaries, curated data,
models, forecast CSVs and artifacts all silently stopped arriving there,
while the runs themselves reported success.

The mount is the only thing missing: the Files API reaches the same Volume
over REST from inside the container. This module carries the outputs
across before the run exits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forecast_engine.s03_storage.volume_sync import (
    SyncTarget,
    VolumeSync,
    sync_outputs_to_volume,
)


class _FakeFiles:
    def __init__(self, fail_for=(), fail_times=0):
        self.uploaded: dict[str, bytes] = {}
        self.attempts: list[str] = []
        self._fail_for = set(fail_for)
        self._fail_times = fail_times
        self._failures: dict[str, int] = {}

    def upload(self, path, contents, overwrite=False):
        self.attempts.append(path)
        if path in self._fail_for:
            seen = self._failures.get(path, 0)
            if seen < self._fail_times:
                self._failures[path] = seen + 1
                raise RuntimeError("volume temporarily unavailable")
        self.uploaded[path] = contents.read()


class _FakeClient:
    def __init__(self, **kwargs):
        self.files = _FakeFiles(**kwargs)


@pytest.fixture
def staged(tmp_path):
    """A container run's outputs, laid out as the engine writes them."""
    run = tmp_path / "workspace" / "runs" / "run-1"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"run_id": "run-1"}))
    (run / "live_status.json").write_text("{}")
    models = tmp_path / "workspace" / "models" / "run-1"
    models.mkdir(parents=True)
    (models / "0_0_model.pkl").write_bytes(b"\x80\x04model")
    forecasts = tmp_path / "workspace" / "forecasts"
    forecasts.mkdir(parents=True)
    (forecasts / "run-1_forecast.csv").write_text("date,value\n2026-01-01,1\n")
    return tmp_path / "workspace"


# --- the copy itself ---------------------------------------------------


def test_a_directory_is_copied_file_by_file_preserving_its_shape(staged):
    client = _FakeClient()
    outcome = VolumeSync(client).run(
        [SyncTarget(staged / "runs" / "run-1", "/Volumes/cat/sch/forecast_files/runs/run-1")]
    )

    assert outcome.ok
    assert set(client.files.uploaded) == {
        "/Volumes/cat/sch/forecast_files/runs/run-1/summary.json",
        "/Volumes/cat/sch/forecast_files/runs/run-1/live_status.json",
    }
    assert json.loads(client.files.uploaded["/Volumes/cat/sch/forecast_files/runs/run-1/summary.json"]) == {
        "run_id": "run-1"
    }


def test_nested_directories_keep_their_relative_paths(tmp_path):
    root = tmp_path / "artifacts" / "run-1"
    (root / "plots").mkdir(parents=True)
    (root / "insights.json").write_text("{}")
    (root / "plots" / "group_a.png").write_bytes(b"png")
    client = _FakeClient()

    VolumeSync(client).run([SyncTarget(root, "/Volumes/cat/sch/artifacts_files/runs/run-1")])

    assert "/Volumes/cat/sch/artifacts_files/runs/run-1/plots/group_a.png" in client.files.uploaded


def test_a_single_file_target_is_copied_to_its_exact_destination(staged):
    client = _FakeClient()
    outcome = VolumeSync(client).run(
        [
            SyncTarget(
                staged / "forecasts" / "run-1_forecast.csv",
                "/Volumes/cat/sch/forecasts_files/runs/run-1_forecast.csv",
            )
        ]
    )

    assert outcome.ok and outcome.files_copied == 1
    assert client.files.uploaded["/Volumes/cat/sch/forecasts_files/runs/run-1_forecast.csv"].startswith(b"date,value")


def test_binary_content_survives_the_copy(staged):
    client = _FakeClient()
    VolumeSync(client).run([SyncTarget(staged / "models" / "run-1", "/Volumes/cat/sch/models_files/runs/run-1")])

    assert client.files.uploaded["/Volumes/cat/sch/models_files/runs/run-1/0_0_model.pkl"] == b"\x80\x04model"


# --- what a run legitimately does not produce --------------------------


def test_a_source_the_run_never_wrote_is_skipped_not_failed(tmp_path):
    """Insights off means no artifacts mirror. That is not a failure."""
    client = _FakeClient()
    outcome = VolumeSync(client).run(
        [SyncTarget(tmp_path / "never-written", "/Volumes/cat/sch/artifacts_files/runs/run-1")]
    )

    assert outcome.ok
    assert outcome.files_copied == 0
    assert outcome.skipped == [str(tmp_path / "never-written")]


# --- failure must be loud ----------------------------------------------


def test_a_transient_failure_is_retried_and_succeeds(staged, monkeypatch):
    monkeypatch.setattr("forecast_engine.s03_storage.volume_sync._RETRY_BACKOFF_SECONDS", 0)
    target = "/Volumes/cat/sch/forecasts_files/runs/run-1_forecast.csv"
    client = _FakeClient(fail_for=[target], fail_times=2)

    outcome = VolumeSync(client).run(
        [SyncTarget(staged / "forecasts" / "run-1_forecast.csv", target)]
    )

    assert outcome.ok and outcome.files_copied == 1
    assert client.files.attempts.count(target) == 3


def test_a_permanent_failure_is_reported_never_swallowed(staged, monkeypatch):
    """The whole point: results that did not reach the volume must say so."""
    monkeypatch.setattr("forecast_engine.s03_storage.volume_sync._RETRY_BACKOFF_SECONDS", 0)
    target = "/Volumes/cat/sch/forecasts_files/runs/run-1_forecast.csv"
    client = _FakeClient(fail_for=[target], fail_times=99)

    outcome = VolumeSync(client).run(
        [SyncTarget(staged / "forecasts" / "run-1_forecast.csv", target)]
    )

    assert not outcome.ok
    assert target in outcome.errors[0]


def test_one_failed_file_does_not_abandon_the_rest(staged, monkeypatch):
    monkeypatch.setattr("forecast_engine.s03_storage.volume_sync._RETRY_BACKOFF_SECONDS", 0)
    doomed = "/Volumes/cat/sch/forecast_files/runs/run-1/summary.json"
    client = _FakeClient(fail_for=[doomed], fail_times=99)

    outcome = VolumeSync(client).run(
        [SyncTarget(staged / "runs" / "run-1", "/Volumes/cat/sch/forecast_files/runs/run-1")]
    )

    assert not outcome.ok
    assert "/Volumes/cat/sch/forecast_files/runs/run-1/live_status.json" in client.files.uploaded


# --- only container runs sync ------------------------------------------


def test_no_volume_sync_block_means_no_copy_and_no_client_is_built():
    """A local run, and any Databricks run not in a container, already
    writes to its final home — building an SDK client would be wrong."""
    assert sync_outputs_to_volume({}) is None
    assert sync_outputs_to_volume(None) is None
    assert sync_outputs_to_volume({"volume_sync": {"targets": []}}) is None


def test_the_backends_payload_shape_drives_the_copy(staged):
    client = _FakeClient()
    payload = {
        "volume_sync": {
            "targets": [
                {
                    "source": str(staged / "runs" / "run-1"),
                    "destination": "/Volumes/cat/sch/forecast_files/runs/run-1",
                }
            ]
        }
    }

    outcome = sync_outputs_to_volume(payload, client)

    assert outcome is not None and outcome.ok and outcome.files_copied == 2
