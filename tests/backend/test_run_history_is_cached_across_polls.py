"""History must not be re-read from MLflow on every poll.

The defect this pins, measured against the deployment's Databricks-hosted
tracking server: one `list_runs()` sweep of fourteen runs cost 39.6s — an
11.3s `get_experiment_by_name` plus 28.3s of paging. `GET /deployments`
called it unconditionally, five screens poll that endpoint every five
seconds, and every route in this API is a sync `def` sharing one threadpool
behind a single gunicorn worker. So the Deployments page failed its own 30s
client timeout ("The request took too long to respond"), and because the
timed-out requests kept running server-side they starved the threadpool and
took every *other* page down with them.

Three things fix it, and all three are pinned here: resolve the experiment
id once, serve the listing from a short-lived cache, and never let a
request thread wait on a sweep that is already in flight.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.config.settings import Settings
from app.orchestration.mlflow_history import (
    DATASET_NAME_TAG,
    HISTORY_COLD_WAIT_SECONDS,
    RUN_ID_TAG,
    MLflowHistoryStore,
)


class _Info:
    def __init__(self, run_id, status="FINISHED", start=1_700_000_000_000, end=1_700_000_060_000):
        self.run_id = run_id
        self.status = status
        self.start_time = start
        self.end_time = end


class _Data:
    def __init__(self, tags):
        self.tags = tags


class _Run:
    def __init__(self, run_id):
        self.info = _Info(f"mlflow-{run_id}")
        self.data = _Data({RUN_ID_TAG: run_id, DATASET_NAME_TAG: "sales.csv"})


class _FakeClient:
    """Counts what it is asked for, and can be made to block or fail."""

    def __init__(self, runs=("run-a", "run-b")):
        self._runs = [_Run(r) for r in runs]
        self.experiment_lookups = 0
        self.searches = 0
        self.gate: threading.Event | None = None
        self.fail = False

    def get_experiment_by_name(self, name):
        self.experiment_lookups += 1
        return type("Exp", (), {"experiment_id": "exp-1"})()

    def search_runs(self, experiment_ids, filter_string="", max_results=None, order_by=None, page_token=None):
        self.searches += 1
        if self.gate is not None:
            self.gate.wait(timeout=10)
        if self.fail:
            raise RuntimeError("tracking server unreachable")
        return list(self._runs)


def _store(client):
    store = MLflowHistoryStore(Settings(execution_mode="local"))
    store._client = client
    return store


def _settled(store, limit=None):
    """Wait for any in-flight sweep to finish, so counts are stable."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with store._lock:
            if not store._listing_refresh:
                return
        time.sleep(0.01)
    raise AssertionError("a history sweep never finished")


# --- the experiment id is resolved once, not per read ------------------


def test_the_experiment_id_is_resolved_once_across_many_history_reads():
    """11.3s cold / 3.1s warm, to re-derive a mapping that cannot change."""
    client = _FakeClient()
    store = _store(client)

    for _ in range(5):
        store.list_runs()
        _settled(store)
        # Expire the listing cache so only the id cache is under test.
        with store._lock:
            store._listing_cache.clear()

    assert client.searches == 5
    assert client.experiment_lookups == 1


def test_a_failed_search_drops_the_cached_experiment_id():
    """Cached wrongness must be self-healing, not permanent."""
    client = _FakeClient()
    store = _store(client)
    store.list_runs()
    _settled(store)
    assert client.experiment_lookups == 1

    client.fail = True
    with store._lock:
        store._listing_cache.clear()
    store.list_runs()
    _settled(store)

    client.fail = False
    with store._lock:
        store._listing_cache.clear()
    store.list_runs()
    _settled(store)

    assert client.experiment_lookups == 2


# --- the listing is cached across polls --------------------------------


def test_repeated_polls_within_the_ttl_read_mlflow_once():
    client = _FakeClient()
    store = _store(client)

    first = store.list_runs()
    _settled(store)
    for _ in range(10):
        assert [r.run_id for r in store.list_runs()] == [r.run_id for r in first]

    assert client.searches == 1


def test_each_limit_is_cached_separately():
    """Estimation asks for a small sample; the deployments list asks for
    everything. One is not a correct answer to the other."""
    client = _FakeClient()
    store = _store(client)

    store.list_runs(limit=200)
    _settled(store)
    store.list_runs(limit=15)
    _settled(store)

    assert client.searches == 2
    assert store.list_runs(limit=200) and store.list_runs(limit=15)
    assert client.searches == 2


# --- a request never waits on a sweep already in flight ----------------


def test_a_stale_listing_is_served_immediately_while_a_sweep_runs():
    """The starvation fix: a poll behind a running sweep answers now."""
    client = _FakeClient()
    store = _store(client)
    store.list_runs()
    _settled(store)

    # Force the cache stale, then make the next sweep hang.
    with store._lock:
        deadline, listings = store._listing_cache[200]
        store._listing_cache[200] = (time.monotonic() - 1, listings)
    client.gate = threading.Event()

    try:
        store.list_runs()  # starts the hanging sweep, returns stale
        started = time.monotonic()
        for _ in range(5):
            assert [r.run_id for r in store.list_runs()] == ["run-a", "run-b"]
        elapsed = time.monotonic() - started

        assert elapsed < 1.0, f"a poll waited {elapsed:.1f}s on an in-flight sweep"
        # Every one of those polls joined the same sweep.
        assert client.searches == 2
    finally:
        client.gate.set()
        _settled(store)


def test_a_cold_read_waits_only_a_bounded_time_for_the_first_sweep():
    """No cache to fall back on, so it waits — but not past the frontend's
    own timeout, which is what turned a slow page into a broken one."""
    client = _FakeClient()
    client.gate = threading.Event()
    store = _store(client)

    try:
        started = time.monotonic()
        assert store.list_runs() == []
        elapsed = time.monotonic() - started

        assert elapsed == pytest.approx(HISTORY_COLD_WAIT_SECONDS, abs=1.0)
    finally:
        client.gate.set()
        _settled(store)


def test_a_fast_store_still_answers_completely_on_a_cold_read():
    """Local development is file-backed and answers in milliseconds; it
    must be entirely unaffected by any of the above."""
    client = _FakeClient()
    store = _store(client)

    assert [r.run_id for r in store.list_runs()] == ["run-a", "run-b"]


# --- failure must not empty a good listing -----------------------------


def test_a_failed_sweep_keeps_serving_the_previous_listing():
    client = _FakeClient()
    store = _store(client)
    store.list_runs()
    _settled(store)

    client.fail = True
    with store._lock:
        deadline, listings = store._listing_cache[200]
        store._listing_cache[200] = (time.monotonic() - 1, listings)

    store.list_runs()
    _settled(store)

    assert [r.run_id for r in store.list_runs()] == ["run-a", "run-b"]


def test_an_unreachable_tracking_server_is_never_cached_as_empty():
    store = MLflowHistoryStore(Settings(execution_mode="local"))
    store._client = None
    store._get_client = lambda: None  # type: ignore[method-assign]

    assert store.list_runs() == []
    with store._lock:
        assert store._listing_cache == {}


# --- the first request must not be the one that pays ------------------


def test_prewarm_fills_the_cache_without_a_request_waiting():
    """Startup absorbs the cold sweep, so the first page load is warm."""
    client = _FakeClient()
    store = _store(client)

    store.prewarm()
    _settled(store)

    started = time.monotonic()
    assert [r.run_id for r in store.list_runs()] == ["run-a", "run-b"]
    assert time.monotonic() - started < 0.5
    assert client.searches == 1


def test_prewarm_returns_immediately_and_never_raises():
    """It runs during application startup; it must not block or fail it."""
    client = _FakeClient()
    client.gate = threading.Event()
    store = _store(client)

    try:
        started = time.monotonic()
        store.prewarm()
        assert time.monotonic() - started < 0.5
    finally:
        client.gate.set()
        _settled(store)


def test_prewarm_does_not_start_a_second_sweep_alongside_a_request():
    client = _FakeClient()
    client.gate = threading.Event()
    store = _store(client)

    try:
        store.prewarm()
        store.prewarm()
        store.list_runs()
        assert client.searches == 1
    finally:
        client.gate.set()
        _settled(store)


def test_the_executor_warms_history_through_the_active_runner():
    """The wiring app startup relies on, pinned end to end."""
    from app.orchestration.executor import PipelineExecutor
    from app.orchestration.local_runner import LocalRunner

    runner = LocalRunner(Settings(execution_mode="local"))
    client = _FakeClient()
    runner._history._client = client

    PipelineExecutor(runner=runner).prewarm()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and client.searches == 0:
        time.sleep(0.01)
    assert client.searches == 1
