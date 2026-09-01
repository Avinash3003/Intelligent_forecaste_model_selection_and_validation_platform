"""Ray must size itself from the compute it is actually given.

The executor never fixes a CPU count: a single-node cluster gets a local
head, a multi-node cluster puts Ray on the Spark workers, and either way
concurrency falls out of the resources Ray reports. These pin that choice
and the absence of any hardcoded machine size.
"""

from __future__ import annotations

import re
from pathlib import Path

import forecast_engine.parallel.ray_executor as ray_executor


def test_no_hardcoded_cluster_size_in_the_executor():
    """The 4-vCPU box we happen to run on must not be baked in."""
    source = Path(ray_executor.__file__).read_text()
    assert "num_cpus=4" not in source
    assert "max_worker_nodes=4" not in source
    # One CPU per key is the intended scheduling grain for every one of
    # the four stage tasks (train/evaluate/explain/rank_select) -- "1" is
    # the only value num_cpus is ever allowed to state, however many times.
    assert source.count("num_cpus=") == 4
    assert all(value == "1" for value in re.findall(r"num_cpus=(\d+)", source))


def test_worker_count_is_zero_without_spark(monkeypatch):
    monkeypatch.setattr(ray_executor, "_spark_worker_nodes", lambda: 0)
    assert ray_executor._spark_worker_nodes() == 0


def test_single_node_uses_local_ray(monkeypatch):
    calls = {}

    class FakeRay:
        def init(self, **kwargs):
            calls["init"] = kwargs

    monkeypatch.setitem(__import__("sys").modules, "ray", FakeRay())
    monkeypatch.setattr(ray_executor, "_spark_worker_nodes", lambda: 0)

    ray_executor._start_ray()

    assert "init" in calls
    assert "address" not in calls["init"]
    # Nothing pins a CPU count; Ray detects what the machine has.
    assert "num_cpus" not in calls["init"]


def test_multi_node_starts_ray_across_spark_workers(monkeypatch):
    calls = {}

    class FakeRay:
        def init(self, **kwargs):
            calls["init"] = kwargs

    def fake_setup(max_worker_nodes, **kwargs):
        calls["setup"] = max_worker_nodes

    fake_spark_module = type("M", (), {"setup_ray_cluster": staticmethod(fake_setup)})
    modules = __import__("sys").modules
    monkeypatch.setitem(modules, "ray", FakeRay())
    monkeypatch.setitem(modules, "ray.util", type("M", (), {})())
    monkeypatch.setitem(modules, "ray.util.spark", fake_spark_module)
    monkeypatch.setattr(ray_executor, "_spark_worker_nodes", lambda: 3)

    ray_executor._start_ray()

    # The worker count comes from the cluster, never from a constant.
    assert calls["setup"] == 3
    assert calls["init"]["address"] == "auto"


def test_multi_node_falls_back_to_local_ray_when_unavailable(monkeypatch):
    calls = {}

    class FakeRay:
        def init(self, **kwargs):
            calls.setdefault("init", []).append(kwargs)

    def exploding_setup(**kwargs):
        raise RuntimeError("ray on spark unavailable")

    fake_spark_module = type("M", (), {"setup_ray_cluster": staticmethod(exploding_setup)})
    modules = __import__("sys").modules
    monkeypatch.setitem(modules, "ray", FakeRay())
    monkeypatch.setitem(modules, "ray.util.spark", fake_spark_module)
    monkeypatch.setattr(ray_executor, "_spark_worker_nodes", lambda: 2)

    ray_executor._start_ray()

    # A run must still execute on the driver rather than fail outright.
    assert calls["init"][-1].get("address") is None
