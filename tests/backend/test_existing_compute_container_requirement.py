"""The compute picker must say why Existing Compute is unavailable before a
user chooses it, not only after they submit and get refused.

`ExistingComputeListResponse.available/message` already existed for "no
clusters available" and "unreachable" -- the container runtime requirement
is reported through the exact same shape, so no frontend change was
needed: StepComputeConfiguration.jsx already renders `result.message`
whenever `result.available` is false.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.services.compute_service import ComputeService


def _settings(**overrides):
    fields = {
        "execution_mode": "databricks",
        "databricks_host": "https://example.invalid",
        "databricks_token": "t",
        "databricks_docker_image_url": "acr.io/forecastiq:1",
    }
    fields.update(overrides)
    return Settings(**fields)


def test_existing_compute_is_reported_unavailable_when_the_container_is_required():
    service = ComputeService(settings=_settings())

    result = service.list_existing_compute()

    assert result.available is False
    assert "New Job Compute" in result.message
    assert result.clusters == []


def test_the_message_explains_why_not_merely_that_it_cannot():
    service = ComputeService(settings=_settings())

    result = service.list_existing_compute()

    assert "container" in result.message.lower()


def test_no_live_cluster_lookup_happens_when_the_requirement_blocks_it_first():
    """The container check must short-circuit before any Databricks call --
    a cluster ID configured for legacy/rollback use must not be probed on
    every picker load once it can never be offered anyway."""
    calls = []

    class _ExplodingClient:
        def __getattr__(self, name):
            calls.append(name)
            raise AssertionError(f"must not touch the Databricks client (.{name})")

    service = ComputeService(settings=_settings(), workspace=_ExplodingClient())

    result = service.list_existing_compute()

    assert result.available is False
    assert calls == []


def test_existing_compute_is_offered_when_the_requirement_is_off():
    """The rollback path: with the requirement disabled, the endpoint
    behaves exactly as it always did -- this test only proves the new
    check does not fire, not the full lookup, which is exercised
    elsewhere against a real cluster fixture."""
    calls = []

    class _StubClusters:
        def list(self):
            calls.append("list")
            raise RuntimeError("simulated: workspace unreachable")

    class _StubClient:
        clusters = _StubClusters()

    service = ComputeService(settings=_settings(databricks_require_container_runtime=False), workspace=_StubClient())

    result = service.list_existing_compute()

    # The lookup was actually attempted (proving the container check did
    # not short-circuit it) and failed for an unrelated reason.
    assert calls == ["list"]
    assert result.available is False
    assert "container" not in result.message.lower()
