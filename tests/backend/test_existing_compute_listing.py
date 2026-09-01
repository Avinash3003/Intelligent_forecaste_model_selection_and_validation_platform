"""The existing-compute picker must offer every all-purpose cluster in the
workspace, not one id fixed in configuration.

Replaces the old single `databricks_existing_cluster_id` setting: with
three (or three hundred) all-purpose clusters in the workspace, all of
them should be selectable, none of them hardcoded anywhere in this
codebase. Job-sourced clusters (a job's own ephemeral compute) and this
service's own validation probes must never appear in that list -- neither
is something a *different* run should be pointed at.

One `clusters.list()` call is the only Databricks round trip this makes,
regardless of how many clusters come back or how many of them qualify --
see the docstring on ComputeService.list_existing_compute.
"""

from __future__ import annotations

from app.services.compute_service import PROBE_NAME_PREFIX, ComputeService


class _State:
    def __init__(self, value):
        self.value = value


class _Source:
    def __init__(self, value):
        self.value = value


class _FakeCluster:
    def __init__(
        self,
        cluster_id,
        cluster_name,
        source="UI",
        state="RUNNING",
        node_type_id="Standard_F4ads_v7",
        spark_version="15.4.x-cpu-ml-scala2.12",
        cluster_cores=4,
        cluster_memory_mb=16384,
        num_workers=0,
        autotermination_minutes=30,
    ):
        self.cluster_id = cluster_id
        self.cluster_name = cluster_name
        self.cluster_source = _Source(source) if source is not None else None
        self.state = _State(state)
        self.node_type_id = node_type_id
        self.spark_version = spark_version
        self.cluster_cores = cluster_cores
        self.cluster_memory_mb = cluster_memory_mb
        self.num_workers = num_workers
        self.autotermination_minutes = autotermination_minutes


def _service(clusters, settings=None):
    service = ComputeService(settings=settings or object(), workspace=object())
    service._client = lambda: type("C", (), {"clusters": type("X", (), {"list": staticmethod(lambda: clusters)})()})()
    return service


def test_every_all_purpose_cluster_is_offered():
    clusters = [
        _FakeCluster("c1", "team-shared-cluster"),
        _FakeCluster("c2", "another-cluster", node_type_id="Standard_E8ads_v7", cluster_cores=8, cluster_memory_mb=32768),
        _FakeCluster("c3", "third-cluster"),
    ]
    result = _service(clusters).list_existing_compute()

    assert result.available is True
    assert {c.cluster_id for c in result.clusters} == {"c1", "c2", "c3"}
    c2 = next(c for c in result.clusters if c.cluster_id == "c2")
    assert c2.cluster_name == "another-cluster"
    assert c2.num_cores == 8
    assert c2.memory_mb == 32768


def test_job_sourced_clusters_are_excluded():
    clusters = [
        _FakeCluster("c1", "shared-cluster", source="UI"),
        _FakeCluster("job1", "job-1234-run-5678", source="JOB"),
    ]
    result = _service(clusters).list_existing_compute()

    assert result.available is True
    assert [c.cluster_id for c in result.clusters] == ["c1"]


def test_this_services_own_validation_probes_are_excluded():
    clusters = [
        _FakeCluster("c1", "shared-cluster"),
        _FakeCluster("probe1", f"{PROBE_NAME_PREFIX}-1700000000", source="UI"),
    ]
    result = _service(clusters).list_existing_compute()

    assert [c.cluster_id for c in result.clusters] == ["c1"]


def test_no_all_purpose_clusters_is_reported_as_unavailable():
    clusters = [_FakeCluster("job1", "job-run", source="JOB")]
    result = _service(clusters).list_existing_compute()

    assert result.available is False
    assert result.clusters == []
    assert "No compatible all-purpose compute" in result.message


def test_a_non_ml_runtime_cluster_is_excluded():
    clusters = [
        _FakeCluster("c1", "shared-cluster"),
        _FakeCluster("c2", "plain-runtime-cluster", spark_version="15.4.x-scala2.12"),
    ]
    result = _service(clusters).list_existing_compute()

    assert [c.cluster_id for c in result.clusters] == ["c1"]


def test_a_run_scoped_cluster_this_app_created_is_excluded():
    """A new_job_compute run's own cluster (see databricks_runner.
    _create_shared_cluster) reports ClusterSource.UI, not JOB — excluded by
    its RUN_CLUSTER_TAG instead."""
    own_run_cluster = _FakeCluster("c2", "forecastiq-dbx-run-abc123")
    own_run_cluster.custom_tags = {"forecastiq_run_id": "dbx-run-abc123"}
    clusters = [_FakeCluster("c1", "shared-cluster"), own_run_cluster]
    result = _service(clusters).list_existing_compute()

    assert [c.cluster_id for c in result.clusters] == ["c1"]


def test_a_listing_failure_is_reported_without_a_raw_exception():
    class _ExplodingClusters:
        def list(self):
            raise RuntimeError("Traceback: connection reset by peer")

    service = ComputeService(settings=object(), workspace=object())
    service._client = lambda: type("C", (), {"clusters": _ExplodingClusters()})()

    result = service.list_existing_compute()

    assert result.available is False
    assert "Traceback" not in result.message
    assert "reached right now" in result.message


def test_exactly_one_workspace_call_regardless_of_cluster_count():
    calls = []

    class _Clusters:
        def list(self):
            calls.append("list")
            return [_FakeCluster(f"c{i}", f"cluster-{i}") for i in range(5)]

    service = ComputeService(settings=object(), workspace=object())
    service._client = lambda: type("C", (), {"clusters": _Clusters()})()

    result = service.list_existing_compute()

    assert calls == ["list"]
    assert len(result.clusters) == 5
