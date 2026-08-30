"""Every persistent pipeline file goes through the one storage adapter.

Before this, a DCS run wrote its outputs to a Workspace folder and a
separate sync step copied them to the volume afterwards — which duplicated
sensitive data under an Analyst-readable path and, when its authentication
broke, silently delivered nothing to the storage account at all.

Now there is a single route decision, made in `core.storage`, and the
writers do not know which side of it they are on. These tests run the real
writers against both routes and assert the bytes arrive either way.
"""

from __future__ import annotations

import io
import json
import pickle

import pandas as pd
import pytest

from forecast_engine.core import storage
from forecast_engine.s01_preprocessing.dataset_loader import DatasetLoader
from forecast_engine.s03_storage.curated_writer import LocalCuratedBackend
from forecast_engine.utils.exceptions import DatasetLoadError

VOLUME = "/Volumes/forecastiq/forecasting"


class _Files:
    def __init__(self):
        self.store: dict[str, bytes] = {}

    def upload(self, path, contents, overwrite=False):
        self.store[path] = contents.read()

    def download(self, path):
        if path not in self.store:
            raise _NotFound(path)
        return type("R", (), {"contents": io.BytesIO(self.store[path])})()

    def get_metadata(self, path):
        if path not in self.store:
            raise _NotFound(path)
        return {}


class _NotFound(Exception):
    pass


_NotFound.__name__ = "NotFound"


class _Client:
    def __init__(self):
        self.files = _Files()


@pytest.fixture
def dcs(monkeypatch):
    """A container: UC Volume paths exist but have no POSIX mount."""
    storage.reset_route_cache()
    storage.reset_client_cache()
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    client = _Client()
    storage.set_files_client(client)
    yield client
    storage.reset_route_cache()
    storage.reset_client_cache()


@pytest.fixture
def existing_compute(monkeypatch):
    """Existing Compute: the mount works, so nothing may touch the API."""
    storage.reset_route_cache()
    storage.reset_client_cache()
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "1")
    client = _Client()
    storage.set_files_client(client)
    yield client
    storage.reset_route_cache()
    storage.reset_client_cache()


FRAME = pd.DataFrame({"date": ["2024-01-01", "2024-02-01"], "sales": [10.0, 12.5]})


# --- reads --------------------------------------------------------------


def test_the_dataset_loads_from_a_volume_with_no_posix_mount(dcs):
    path = f"{VOLUME}/forecast_files/runs/r1/sales.csv"
    dcs.files.store[path] = FRAME.to_csv(index=False).encode()

    loaded = DatasetLoader().load(path)

    assert list(loaded.columns) == ["date", "sales"]
    assert len(loaded) == 2


def test_the_dataset_still_loads_from_an_ordinary_path(existing_compute, tmp_path):
    path = tmp_path / "sales.csv"
    FRAME.to_csv(path, index=False)

    loaded = DatasetLoader().load(path)

    assert len(loaded) == 2
    assert existing_compute.files.store == {}, "POSIX route must not use the API"


def test_a_dataset_that_is_genuinely_absent_is_reported_as_missing(dcs):
    with pytest.raises(DatasetLoadError, match="could not be found"):
        DatasetLoader().load(f"{VOLUME}/forecast_files/runs/r1/nope.csv")


# --- writes -------------------------------------------------------------


def test_curated_output_is_written_to_the_volume(dcs):
    writer = LocalCuratedBackend(f"{VOLUME}/curated_files/runs")

    uri = writer.write(FRAME, "r1/curated.csv", "csv")

    written = dcs.files.store[f"{VOLUME}/curated_files/runs/r1/curated.csv"]
    assert pd.read_csv(io.BytesIO(written)).equals(FRAME)
    assert "curated_files" in uri


def test_curated_parquet_survives_the_buffer_round_trip(dcs):
    writer = LocalCuratedBackend(f"{VOLUME}/curated_files/runs")

    writer.write(FRAME, "r1/curated.parquet", "parquet")

    written = dcs.files.store[f"{VOLUME}/curated_files/runs/r1/curated.parquet"]
    assert pd.read_parquet(io.BytesIO(written)).equals(FRAME)


def test_a_model_binary_survives_the_volume_round_trip(dcs):
    """Pickles are the one payload a text-mode slip would silently corrupt."""
    path = f"{VOLUME}/models_files/runs/r1/0_0_model.pkl"
    model = {"weights": [0.1, 0.2], "blob": b"\x80\x04\xff\x00"}

    storage.write_bytes(path, pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))

    assert pickle.loads(storage.read_bytes(path)) == model


def test_the_summary_is_written_to_the_volume(dcs):
    path = f"{VOLUME}/forecast_files/runs/r1/summary.json"

    storage.write_text(path, json.dumps({"run_id": "r1", "stages": []}))

    assert json.loads(dcs.files.store[path])["run_id"] == "r1"


# --- live status --------------------------------------------------------


def test_live_status_uses_a_single_put_when_there_is_no_mount(dcs):
    """No rename exists on the Files API. A single-request PUT replaces the
    object wholesale, so a poller never sees half a document."""
    from forecast_engine.core.live_status import LiveStatusWriter

    path = f"{VOLUME}/forecast_files/runs/r1/live_status.json"
    writer = LiveStatusWriter(path)
    context = type("C", (), {
        "run_id": "r1",
        "started_at": type("D", (), {"isoformat": lambda self, timespec=None: "2026-01-01T00:00:00"})(),
        "stages": [],
    })()

    writer(context)

    assert json.loads(dcs.files.store[path])["run_id"] == "r1"


def test_live_status_keeps_atomic_replace_where_posix_works(existing_compute, tmp_path):
    from forecast_engine.core.live_status import LiveStatusWriter

    path = tmp_path / "live_status.json"
    writer = LiveStatusWriter(path)
    context = type("C", (), {
        "run_id": "r2",
        "started_at": type("D", (), {"isoformat": lambda self, timespec=None: "2026-01-01T00:00:00"})(),
        "stages": [],
    })()

    writer(context)

    assert json.loads(path.read_text())["run_id"] == "r2"
    assert existing_compute.files.store == {}, "POSIX route must not use the API"
    assert not list(tmp_path.glob(".live_status_*")), "temp file must be cleaned up"
