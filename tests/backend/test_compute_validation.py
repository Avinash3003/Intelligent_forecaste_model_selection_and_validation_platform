"""Compute validation must never report a configuration valid on its own.

These pin the parts that decide what the user is told: the quota
arithmetic, the metadata gate, and the mapping from a Databricks failure
to one short sentence. The create probe itself is exercised against the
real workspace, not here.
"""

from __future__ import annotations

import time

import pytest

from app.schemas.compute import JobComputeConfig
from app.services import compute_service
from app.services.compute_service import (
    GENERIC_FAILURE,
    ComputeService,
    message_for_api_error,
    message_for_termination,
)

RUNTIME = "15.4.x-cpu-ml-scala2.12"


def _service(node):
    """A service whose live catalog lookup returns one known node."""
    service = ComputeService(settings=object(), workspace=object())
    service._node_info = lambda node_type_id: node if node and node["node_type_id"] == node_type_id else None
    return service


def _node(node_type_id="Standard_F4ads_v7", cores=4, quota=4, status=None):
    info = {"available_core_quota": quota}
    if status:
        info["status"] = [status]
    return {"node_type_id": node_type_id, "num_cores": cores, "node_info": info}


# ---- requested cores ------------------------------------------------


@pytest.mark.parametrize(
    "config, cores_per_node, expected",
    [
        (dict(num_workers=0), 4, 4),  # driver only
        (dict(num_workers=2), 4, 12),  # driver + 2 workers
        (dict(autoscale=True, min_workers=1, max_workers=3), 4, 16),  # bounded by max
        (dict(num_workers=1), 8, 16),
    ],
)
def test_requested_cores_counts_the_driver(config, cores_per_node, expected):
    job = JobComputeConfig(node_type_id="n", runtime_key=RUNTIME, **config)
    assert job.requested_cores(cores_per_node) == expected


def test_autoscale_bounds_are_rejected_when_inverted():
    with pytest.raises(ValueError):
        JobComputeConfig(
            node_type_id="n", runtime_key=RUNTIME, autoscale=True, min_workers=5, max_workers=2
        )


def test_a_single_node_request_is_not_rejected_for_its_unused_autoscale_bounds():
    """The reported bug: choosing 0 workers with autoscaling off failed with
    "Please check these fields and try again: max_workers".

    The UI hides the min/max inputs when autoscaling is off, but still sends
    them, and its number input reports an emptied box as 0. Validating a
    bound the user cannot see, for a mode it does not apply to, blocked a
    valid single-node run."""
    config = JobComputeConfig(
        node_type_id="n", runtime_key=RUNTIME, autoscale=False, num_workers=0, max_workers=0
    )

    assert config.num_workers == 0
    # One node, no workers -- the bounds contribute nothing.
    assert config.requested_cores(4) == 4


def test_autoscaling_still_requires_a_real_upper_bound():
    """Relaxing the field-level floor must not let an autoscaling cluster
    through with no ceiling to scale to."""
    with pytest.raises(ValueError):
        JobComputeConfig(
            node_type_id="n", runtime_key=RUNTIME, autoscale=True, min_workers=0, max_workers=0
        )


# ---- stage 1 --------------------------------------------------------


def test_configuration_within_quota_passes_metadata():
    service = _service(_node(quota=4))
    result = service.validate(
        JobComputeConfig(node_type_id="Standard_F4ads_v7", runtime_key=RUNTIME, num_workers=0),
        quick=True,
    )
    assert result.valid
    assert result.stage == "metadata"


def test_configuration_over_quota_is_rejected_with_the_numbers():
    service = _service(_node(quota=4))
    result = service.validate(
        JobComputeConfig(node_type_id="Standard_F4ads_v7", runtime_key=RUNTIME, num_workers=3),
        quick=True,
    )
    assert not result.valid
    assert "16 vCPUs" in result.message and "only 4" in result.message


def test_node_type_unavailable_on_the_subscription_is_rejected():
    service = _service(_node(status="NotEnabledOnSubscription"))
    result = service.validate(
        JobComputeConfig(node_type_id="Standard_F4ads_v7", runtime_key=RUNTIME), quick=True
    )
    assert not result.valid
    assert "not enabled on this subscription" in result.message


def test_unknown_node_type_is_rejected():
    service = _service(_node())
    result = service.validate(
        JobComputeConfig(node_type_id="Standard_NOPE_v9", runtime_key=RUNTIME), quick=True
    )
    assert not result.valid
    assert "not offered by your workspace" in result.message


def test_a_node_type_reporting_no_quota_is_not_blocked_on_quota():
    """The workspace reports no quota for some node types; absence of a
    number must not be read as a quota of zero."""
    service = _service(_node(quota=None))
    result = service.validate(
        JobComputeConfig(node_type_id="Standard_F4ads_v7", runtime_key=RUNTIME, num_workers=2),
        quick=True,
    )
    assert result.valid


def test_metadata_failure_skips_the_create_probe():
    service = _service(_node(quota=4))
    service._validate_by_create_probe = lambda config: pytest.fail("probe must not run")
    result = service.validate(
        JobComputeConfig(node_type_id="Standard_F4ads_v7", runtime_key=RUNTIME, num_workers=9)
    )
    assert not result.valid


# ---- error mapping --------------------------------------------------


@pytest.mark.parametrize(
    "code, expected",
    [
        ("AZURE_QUOTA_EXCEEDED_EXCEPTION", "quota"),
        ("CLOUD_PROVIDER_RESOURCE_STOCKOUT", "not available"),
        ("UNSUPPORTED_INSTANCE_TYPE", "not available"),
        ("INVALID_ARGUMENT", "not supported"),
    ],
)
def test_termination_codes_map_to_short_sentences(code, expected):
    message = message_for_termination(code)
    assert expected in message.lower()
    assert "\n" not in message


def test_unknown_termination_code_falls_back_to_the_generic_message():
    assert message_for_termination("SOMETHING_NEW_2031") == GENERIC_FAILURE


@pytest.mark.parametrize(
    "error, expected",
    [
        (Exception("QUOTA_EXCEEDED: not enough cores"), "quota"),
        (Exception("User does not have permission to create clusters"), "permission"),
        (Exception("cluster policy 123 forbids this"), "policy"),
        (Exception("NO_ISOLATION or custom access modes are not allowed"), "not supported"),
    ],
)
def test_api_errors_map_to_short_sentences(error, expected):
    assert expected in message_for_api_error(error).lower()


def test_raw_databricks_detail_never_reaches_the_message():
    noisy = Exception("Traceback (most recent call last): RESOURCE_EXHAUSTED at 0x7f\n  File x")
    message = message_for_api_error(noisy)
    assert "Traceback" not in message
    assert "\n" not in message


# ---- existing compute validation ------------------------------------


class _FakeReason:
    def __init__(self, code=None, kind="SUCCESS"):
        self.code = type("C", (), {"value": code})() if code else None
        self.type = type("T", (), {"value": kind})()


class _FakeCluster:
    def __init__(self, state="RUNNING", single_user="me@example.com", use_ml_runtime=True,
                 spark_version="15.4.x-scala2.12", reason=None):
        self.state = type("S", (), {"value": state})()
        self.single_user_name = single_user
        self.use_ml_runtime = use_ml_runtime
        self.spark_version = spark_version
        self.termination_reason = reason
        self.cluster_name = "forecastiq-ray-dev"
        self.num_workers = 0


def _existing_service(cluster=None, error=None, levels=("CAN_MANAGE",), caller="me@example.com"):
    settings = type("S", (), {"databricks_existing_cluster_id": "cid"})()
    service = ComputeService(settings=settings, workspace=object())

    def get(cluster_id):
        if error:
            raise error
        return cluster

    service._client = lambda: type("C", (), {"clusters": type("X", (), {"get": staticmethod(get)})()})()
    service._current_user_name = lambda: caller
    service._permissions_for = lambda cluster_id: set(levels) if levels is not None else None
    return service


def test_existing_running_cluster_is_valid():
    result = _existing_service(_FakeCluster(state="RUNNING")).validate_existing_compute()
    assert result.valid
    assert result.state == "RUNNING"
    assert not result.starts_on_demand
    assert "ready to run" in result.message


def test_existing_terminated_cluster_is_valid_and_starts_on_demand():
    cluster = _FakeCluster(state="TERMINATED", reason=_FakeReason("INACTIVITY", "SUCCESS"))
    result = _existing_service(cluster).validate_existing_compute()
    assert result.valid
    assert result.starts_on_demand
    assert "stopped but will start" in result.message


# A stopped cluster must be answered from metadata alone. Starting it, or
# waiting for it to reach RUNNING, would hold the wizard for minutes on the
# exact step the user is trying to get past -- Databricks starts the cluster
# when the run is submitted, which is soon enough.
def test_terminated_validation_never_starts_or_waits_for_the_cluster(monkeypatch):
    calls = []

    def forbidden(name):
        def _call(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"validation must not call clusters.{name}()")
        return _call

    class _Clusters:
        def __init__(self, cluster):
            self._cluster = cluster
            self.reads = 0

        def get(self, cluster_id):
            self.reads += 1
            return self._cluster

        start = forbidden("start")
        restart = forbidden("restart")
        ensure_cluster_is_running = forbidden("ensure_cluster_is_running")
        wait_get_cluster_running = forbidden("wait_get_cluster_running")

    monkeypatch.setattr(
        compute_service.time,
        "sleep",
        lambda *_: (_ for _ in ()).throw(AssertionError("validation must not sleep")),
    )

    clusters = _Clusters(_FakeCluster(state="TERMINATED", reason=_FakeReason("INACTIVITY", "SUCCESS")))
    settings = type("S", (), {"databricks_existing_cluster_id": "cid"})()
    service = ComputeService(settings=settings, workspace=object())
    service._client = lambda: type("C", (), {"clusters": clusters})()
    service._current_user_name = lambda: "me@example.com"
    service._permissions_for = lambda cluster_id: {"CAN_MANAGE"}

    started = time.monotonic()
    result = service.validate_existing_compute()
    elapsed = time.monotonic() - started

    assert result.valid
    assert result.state == "TERMINATED"
    assert result.starts_on_demand
    assert calls == []
    # One read of the cluster, and no polling loop behind it.
    assert clusters.reads == 1
    assert elapsed < 1.0


def test_existing_cluster_terminated_by_failure_is_invalid():
    cluster = _FakeCluster(state="TERMINATED",
                           reason=_FakeReason("AZURE_QUOTA_EXCEEDED_EXCEPTION", "CLOUD_FAILURE"))
    result = _existing_service(cluster).validate_existing_compute()
    assert not result.valid
    assert "quota" in result.message.lower()


def test_existing_cluster_not_found_is_invalid():
    service = _existing_service(error=Exception("Cluster cid does not exist"))
    result = service.validate_existing_compute()
    assert not result.valid
    assert "could not be found" in result.message


def test_existing_cluster_permission_denied_is_invalid():
    service = _existing_service(error=Exception("User is not authorized to access cluster"))
    result = service.validate_existing_compute()
    assert not result.valid
    assert "not accessible" in result.message


def test_existing_cluster_without_attach_permission_is_invalid():
    service = _existing_service(_FakeCluster(), levels=("CAN_VIEW",))
    result = service.validate_existing_compute()
    assert not result.valid
    assert "permissions" in result.message


def test_stopped_cluster_without_restart_permission_is_invalid():
    cluster = _FakeCluster(state="TERMINATED", reason=_FakeReason("INACTIVITY", "SUCCESS"))
    result = _existing_service(cluster, levels=("CAN_ATTACH_TO",)).validate_existing_compute()
    assert not result.valid
    assert "cannot be started" in result.message


def test_cluster_reserved_for_another_user_is_invalid():
    service = _existing_service(_FakeCluster(single_user="someone.else@example.com"))
    result = service.validate_existing_compute()
    assert not result.valid
    assert "reserved for another user" in result.message


def test_cluster_without_ml_runtime_is_invalid():
    cluster = _FakeCluster(use_ml_runtime=False, spark_version="15.4.x-scala2.12")
    result = _existing_service(cluster).validate_existing_compute()
    assert not result.valid
    assert "machine learning runtime" in result.message


def test_ml_runtime_detected_from_the_version_string():
    """Some clusters carry the ML stack via the version name instead of the flag."""
    cluster = _FakeCluster(use_ml_runtime=False, spark_version="15.4.x-cpu-ml-scala2.12")
    assert _existing_service(cluster).validate_existing_compute().valid


def test_terminating_cluster_is_invalid():
    result = _existing_service(_FakeCluster(state="TERMINATING")).validate_existing_compute()
    assert not result.valid


def test_no_configured_cluster_is_invalid():
    settings = type("S", (), {"databricks_existing_cluster_id": ""})()
    service = ComputeService(settings=settings, workspace=object())
    result = service.validate_existing_compute()
    assert not result.valid
    assert "No existing compute is configured" in result.message


def test_existing_validation_never_fetches_the_node_catalog():
    """The catalog costs seconds; the existing-compute check must not use it."""
    service = _existing_service(_FakeCluster())
    service._node_info = lambda node_type_id: pytest.fail("node catalog must not be read")
    assert service.validate_existing_compute().valid


def test_existing_validation_messages_stay_short_and_clean():
    service = _existing_service(error=Exception("Traceback:\n  RESOURCE_EXHAUSTED at 0x7f"))
    message = service.validate_existing_compute().message
    assert "\n" not in message and "Traceback" not in message
