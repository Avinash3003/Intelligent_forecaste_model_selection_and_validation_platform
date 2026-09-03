"""The node catalog's `available_core_quota` is not static configuration —
it is the subscription's remaining headroom, which falls while another
cluster holds a family's cores and recovers once that cluster terminates.

The defect this pins: `_ensure_catalog` fetched once per service instance
and never again. A validation that read "0 vCPUs available" because
something else was briefly using the quota kept failing for the rest of
the process's life, long after that quota was free again — reported live
as `Standard_E4ads_v7` failing with "needs 4 vCPUs but only 0 are
available" from a long-running backend that had not refetched since.
"""

from __future__ import annotations

import time

from app.services.compute_service import ComputeService, _NODE_CATALOG_TTL_SECONDS


class _FakeApiClient:
    """Scripts `list-node-types` responses, one per call, so a test can
    prove exactly how many times the workspace was actually asked."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def do(self, method, path):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _service(api_client):
    service = ComputeService(settings=object(), workspace=object())
    service._client = lambda: type("C", (), {"api_client": api_client})()
    return service


def _catalog_payload(quota):
    return {"node_types": [{"node_type_id": "Standard_E4ads_v7", "num_cores": 4.0, "node_info": {"available_core_quota": quota}}]}


def test_a_fresh_catalog_is_reused_without_refetching():
    api = _FakeApiClient([_catalog_payload(4.0)])
    service = _service(api)

    service._ensure_catalog()
    service._ensure_catalog()
    service._ensure_catalog()

    assert api.calls == 1


def test_an_expired_catalog_is_refetched_and_the_new_quota_wins():
    """The exact production case: quota reads 0 while something else holds
    it, then frees up — the next validation must see the recovered number,
    not the number cached when it was exhausted."""
    api = _FakeApiClient([_catalog_payload(0.0), _catalog_payload(4.0)])
    service = _service(api)

    first = service._ensure_catalog()
    assert first["Standard_E4ads_v7"]["node_info"]["available_core_quota"] == 0.0

    # Simulate the TTL having elapsed without a real sleep.
    service._node_catalog_fetched_at -= _NODE_CATALOG_TTL_SECONDS + 1

    second = service._ensure_catalog()
    assert second["Standard_E4ads_v7"]["node_info"]["available_core_quota"] == 4.0
    assert api.calls == 2


def test_a_refresh_failure_keeps_serving_the_last_known_catalog():
    """A transient blip refetching must not turn a working catalog into no
    catalog at all — every validation would fail for no reason."""
    api = _FakeApiClient([_catalog_payload(4.0), RuntimeError("timeout")])
    service = _service(api)

    service._ensure_catalog()
    service._node_catalog_fetched_at -= _NODE_CATALOG_TTL_SECONDS + 1

    result = service._ensure_catalog()

    assert result["Standard_E4ads_v7"]["node_info"]["available_core_quota"] == 4.0
    assert api.calls == 2


def test_the_first_ever_fetch_failing_reports_no_catalog():
    api = _FakeApiClient([RuntimeError("timeout")])
    service = _service(api)

    assert service._ensure_catalog() is None


def test_fetch_timestamp_only_advances_on_a_successful_fetch():
    api = _FakeApiClient([_catalog_payload(4.0), RuntimeError("timeout")])
    service = _service(api)

    service._ensure_catalog()
    stamp_after_success = service._node_catalog_fetched_at
    service._node_catalog_fetched_at -= _NODE_CATALOG_TTL_SECONDS + 1
    stale_stamp = service._node_catalog_fetched_at

    service._ensure_catalog()  # this fetch raises

    assert service._node_catalog_fetched_at == stale_stamp
    assert stale_stamp != stamp_after_success
