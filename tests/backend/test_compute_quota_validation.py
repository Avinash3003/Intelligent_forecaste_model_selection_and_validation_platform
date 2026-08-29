"""A configuration that cannot start must be rejected before it is run.

The defect this pins: the quota guard read `available_core_quota` off the
selected node type and skipped itself entirely when that field was absent —
`if quota is not None`. This workspace reports a quota for
Standard_F4ads_v7, Standard_E4ads_v7 and Standard_F8ads_v7, and none at all
for Standard_DC4as_v5, which happens to be the default preset. So the size
most users pick was the one size never checked.

A three-worker Standard_DC4as_v5 job asks for 16 vCPUs against a regional
limit of 4. It validated clean, the run was submitted, and Databricks
terminated the cluster with AZURE_QUOTA_EXCEEDED_EXCEPTION after the user
had already waited for compute to come up.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.schemas.compute import JobComputeConfig
from app.services.compute_service import ComputeService


def _service(catalog):
    service = ComputeService()
    service._node_catalog = catalog
    return service


def _node(cores, quota=None, status=None):
    info = {}
    if quota is not None:
        info["available_core_quota"] = quota
    if status is not None:
        info["status"] = status
    return {"num_cores": cores, "node_info": info}


# The exact catalog this workspace returns.
WORKSPACE = {
    "Standard_DC4as_v5": _node(4.0),                 # reports no quota
    "Standard_F4ads_v7": _node(4.0, quota=4.0),
    "Standard_E4ads_v7": _node(4.0, quota=4.0),
    "Standard_F8ads_v7": _node(8.0, quota=4.0),
}


def _config(node_type, workers, autoscale=False, max_workers=2):
    return JobComputeConfig(
        node_type_id=node_type,
        runtime_key="15.4.x-scala2.12",
        autoscale=autoscale,
        num_workers=workers,
        max_workers=max_workers,
    )


# --- the bug ----------------------------------------------------------


def test_the_default_node_type_is_checked_even_though_it_reports_no_quota():
    """3 workers + driver = 4 nodes x 4 cores = 16 vCPUs, limit 4."""
    result = _service(WORKSPACE)._validate_against_catalog(_config("Standard_DC4as_v5", 3))

    assert not result.valid
    assert "16" in result.message and "4" in result.message


def test_the_message_names_both_numbers_so_the_user_can_act():
    result = _service(WORKSPACE)._validate_against_catalog(_config("Standard_DC4as_v5", 3))

    assert result.message == (
        "Unable to create compute: this configuration needs 16 vCPUs but only 4 are available."
    )


def test_a_single_node_run_on_that_same_type_still_validates():
    """The fix must not block what actually fits — and does run today."""
    result = _service(WORKSPACE)._validate_against_catalog(_config("Standard_DC4as_v5", 0))

    assert result.valid


# --- unchanged behaviour ----------------------------------------------


def test_a_node_type_with_its_own_quota_still_uses_its_own():
    result = _service(WORKSPACE)._validate_against_catalog(_config("Standard_F4ads_v7", 3))

    assert not result.valid


def test_autoscale_is_measured_at_its_ceiling_not_its_floor():
    result = _service(WORKSPACE)._validate_against_catalog(
        _config("Standard_DC4as_v5", 0, autoscale=True, max_workers=3)
    )

    assert not result.valid


def test_nothing_is_claimed_when_the_whole_catalog_reports_no_quota():
    """No data is not the same as no limit — but inventing one would be
    worse. This leaves the caller exactly where it was."""
    catalog = {"Standard_DC4as_v5": _node(4.0)}

    result = _service(catalog)._validate_against_catalog(_config("Standard_DC4as_v5", 3))

    assert result.valid


def test_the_ceiling_is_the_modal_quota_not_an_outlier():
    """Azure's regional limit binds nearly every family, so the figure most
    node types report is the limit. This workspace: 253 types at 4.0, two
    GPU types at 6.0, twenty disabled types at 0.0."""
    catalog = {f"ordinary-{i}": _node(4.0, quota=4.0) for i in range(253)}
    catalog.update(
        {
            "Standard_NC12": _node(12.0, quota=6.0),
            "Standard_NC24": _node(24.0, quota=6.0),
            "quiet": _node(4.0),
        }
    )
    catalog.update({f"disabled-{i}": _node(4.0, quota=0.0) for i in range(20)})

    assert _service(catalog)._catalog_core_quota() == 4.0


def test_a_disabled_familys_zero_never_becomes_the_ceiling():
    """A zero says the family is unavailable — the status check's job — not
    that the subscription has no cores."""
    catalog = {
        "quiet": _node(4.0),
        "live": _node(4.0, quota=4.0),
        "disabled": _node(4.0, quota=0.0),
    }
    service = _service(catalog)

    assert service._catalog_core_quota() == 4.0
    assert service._validate_against_catalog(_config("quiet", 0)).valid


def test_a_disabled_node_type_is_still_rejected_before_any_quota_maths():
    catalog = {"blocked": _node(4.0, status=["NotEnabledOnSubscription"])}

    result = _service(catalog)._validate_against_catalog(_config("blocked", 0))

    assert not result.valid
    assert "not enabled on this subscription" in result.message


# --- the first user must not pay for the catalog -----------------------


def test_prewarm_loads_the_catalog_without_a_request_waiting():
    """Validation answers in microseconds *from* the catalog; fetching it
    measured 5.13s. Startup absorbs that, so nobody waits on it."""
    service = ComputeService()
    fetched = threading.Event()

    def _fetch():
        fetched.set()
        return {"Standard_DC4as_v5": _node(4.0, quota=4.0)}

    service._ensure_catalog = _fetch  # type: ignore[method-assign]
    service.prewarm()

    assert fetched.wait(timeout=5)


def test_prewarm_returns_immediately_and_never_raises():
    """It runs during application startup; it must not block or fail it."""
    service = ComputeService()
    release = threading.Event()

    def _slow():
        release.wait(timeout=10)
        return {}

    service._ensure_catalog = _slow  # type: ignore[method-assign]
    try:
        started = time.monotonic()
        service.prewarm()
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
