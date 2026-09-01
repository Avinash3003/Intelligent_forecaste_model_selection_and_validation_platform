"""Container-only execution, when a deployment opts into it.

`databricks_require_container_runtime` is off by default -- only TFT
genuinely needs the image, and `unsupported_models` refuses that one model
on its own (see test_container_only_models.py). A deployment that wants
every run on the container turns this on, and Existing Compute is then
refused with a reason rather than silently executing on a runtime that
does not carry every model's dependencies.

These tests therefore enable it explicitly: the default path is covered by
test_existing_compute_stays_available_by_default below.
"""

from __future__ import annotations

import pytest

from app.config.model_availability import compute_rejection_reason, container_runtime_required
from app.config.settings import Settings
from app.schemas.compute import ComputeSelection, JobComputeConfig
from app.services.deployment_service import UnsupportedComputeError, build_execution_request

EXISTING = ComputeSelection(mode="existing_compute", cluster_id="test-cluster")
JOB = ComputeSelection(
    mode="new_job_compute",
    job_compute=JobComputeConfig(node_type_id="Standard_DS3_v2", runtime_key="15.4.x-cpu-ml-scala2.12"),
)


def _settings(**overrides):
    fields = {
        "execution_mode": "databricks",
        "databricks_host": "https://example.invalid",
        "databricks_token": "t",
        "databricks_docker_image_url": "acr.io/forecastiq:1",
        # Opt in: this file is about what the requirement does once on.
        "databricks_require_container_runtime": True,
    }
    fields.update(overrides)
    return Settings(**fields)


class _Principal:
    subject = "u1"
    display_name = "Test User"
    email = "u1@example.com"


def _request(compute, dataset_path):
    from app.schemas.deployment import DeploymentRequest

    return DeploymentRequest(
        file_id="f1",
        dataset_name="sales.csv",
        metadata={
            "date_column": "date",
            "target_column": "sales",
            "key_columns": ["store"],
            "feature_columns": [],
        },
        compute=compute,
    )


# --- the switch itself ---------------------------------------------------


def test_the_requirement_is_active_once_an_image_is_configured():
    assert container_runtime_required(_settings())


def test_the_requirement_is_inactive_without_an_image():
    assert not container_runtime_required(_settings(databricks_docker_image_url=""))


def test_the_requirement_can_be_turned_off_for_rollback():
    """The rollback path: flip one setting, Existing Compute works again."""
    assert not container_runtime_required(_settings(databricks_require_container_runtime=False))


# --- what each compute selection gets told -------------------------------


def test_existing_compute_is_rejected_with_a_reason():
    reason = compute_rejection_reason(EXISTING, _settings())

    assert reason is not None
    assert "New Job Compute" in reason


def test_new_job_compute_is_accepted():
    assert compute_rejection_reason(JOB, _settings()) is None


def test_existing_compute_is_accepted_when_the_requirement_is_off():
    assert compute_rejection_reason(EXISTING, _settings(databricks_require_container_runtime=False)) is None


# --- the deploy path actually refuses, not just the helper function -----


def test_build_execution_request_refuses_existing_compute(tmp_path, monkeypatch):
    """`build_execution_request` reads the process-global settings, so the
    active configuration is what a real request would see -- injected here
    the same way, not through a Settings instance the function never uses."""
    import app.services.deployment_service as deployment_service_module

    monkeypatch.setattr(deployment_service_module, "get_settings", _settings)
    dataset = tmp_path / "sales.csv"
    dataset.write_text("date,store,sales\n2024-01-01,1,10\n")

    with pytest.raises(UnsupportedComputeError, match="New Job Compute"):
        build_execution_request(_request(EXISTING, dataset), dataset, _Principal())


def test_build_execution_request_accepts_new_job_compute(tmp_path, monkeypatch):
    import app.services.deployment_service as deployment_service_module

    monkeypatch.setattr(deployment_service_module, "get_settings", _settings)
    dataset = tmp_path / "sales.csv"
    dataset.write_text("date,store,sales\n2024-01-01,1,10\n")

    req = build_execution_request(_request(JOB, dataset), dataset, _Principal())

    assert req.compute.mode == "new_job_compute"


# --- the default ------------------------------------------------------
#
# This shipped defaulting to True, which disabled Existing Compute in every
# deployment that had an image configured -- including the webapp, where the
# picker showed only a "legacy infrastructure" refusal. Nothing about the
# runtime gap justified that breadth: TFT alone needs the image.


def test_existing_compute_stays_available_by_default():
    """A configured image must not, on its own, refuse Existing Compute."""
    default = _settings(databricks_require_container_runtime=False)

    assert container_runtime_required(default) is False
    assert compute_rejection_reason(EXISTING, default) is None


def test_the_requirement_is_off_unless_a_deployment_asks_for_it():
    settings = Settings(
        execution_mode="databricks",
        databricks_host="https://example.invalid",
        databricks_token="t",
        databricks_docker_image_url="acr.io/forecastiq:1",
    )

    assert settings.databricks_require_container_runtime is False
    assert compute_rejection_reason(EXISTING, settings) is None


# --- an unknown runtime fails before a cluster boots ------------------
#
# Databricks only rejects an unknown spark version when it starts the
# cluster: a typo ("15.4" for "15.4.x-cpu-ml-scala2.12") surfaced as
# "Invalid spark version" about five minutes into a doomed run.


def test_a_runtime_this_platform_does_not_offer_is_refused(tmp_path, monkeypatch):
    import app.services.deployment_service as deployment_service_module

    monkeypatch.setattr(deployment_service_module, "get_settings", _settings)
    dataset = tmp_path / "sales.csv"
    dataset.write_text("date,store,sales\n2024-01-01,1,10\n")
    typo = ComputeSelection(
        mode="new_job_compute",
        job_compute=JobComputeConfig(node_type_id="Standard_DC4as_v5", runtime_key="15.4"),
    )

    with pytest.raises(UnsupportedComputeError, match="not one ForecastIQ offers"):
        build_execution_request(_request(typo, dataset), dataset, _Principal())


def test_the_refusal_names_the_runtimes_that_would_work(tmp_path, monkeypatch):
    from app.config.compute_presets import RUNTIME_PRESETS

    import app.services.deployment_service as deployment_service_module

    monkeypatch.setattr(deployment_service_module, "get_settings", _settings)
    dataset = tmp_path / "sales.csv"
    dataset.write_text("date,store,sales\n2024-01-01,1,10\n")
    typo = ComputeSelection(
        mode="new_job_compute",
        job_compute=JobComputeConfig(node_type_id="Standard_DC4as_v5", runtime_key="nonsense"),
    )

    with pytest.raises(UnsupportedComputeError) as raised:
        build_execution_request(_request(typo, dataset), dataset, _Principal())

    for runtime in RUNTIME_PRESETS:
        assert runtime.key in str(raised.value)


def test_every_offered_runtime_is_accepted(tmp_path, monkeypatch):
    """Guards the reverse mistake: a validator stricter than the picker."""
    from app.config.compute_presets import RUNTIME_PRESETS

    import app.services.deployment_service as deployment_service_module

    monkeypatch.setattr(deployment_service_module, "get_settings", _settings)
    dataset = tmp_path / "sales.csv"
    dataset.write_text("date,store,sales\n2024-01-01,1,10\n")

    for runtime in RUNTIME_PRESETS:
        offered = ComputeSelection(
            mode="new_job_compute",
            job_compute=JobComputeConfig(node_type_id="Standard_DC4as_v5", runtime_key=runtime.key),
        )
        assert build_execution_request(_request(offered, dataset), dataset, _Principal())
