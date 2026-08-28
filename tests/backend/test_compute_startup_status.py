"""A booting cluster must not look like a stuck application.

An existing Databricks cluster that is TERMINATED starts on demand when the
run is submitted, so for the first few minutes the run is real but the
engine has not begun. The trail rendered seven Pending phases and nothing
else, which reads as "nothing is happening".

These pin the fix and, just as importantly, its boundary: the seven display
phases are the forecast engine's own stages, and compute startup must never
become an eighth one.
"""

from __future__ import annotations

from app.orchestration.schemas import JobStatus, RunListing
from app.services.deployment_service import DeploymentService
from app.services.pipeline_stages import PHASE_LABELS


def _listing(
    stages: list[dict] | None = None,
    status: JobStatus = JobStatus.PENDING,
    backend: str = "databricks",
    error: str | None = None,
) -> RunListing:
    return RunListing(
        run_id="dbx-run-test",
        dataset_name="store_item_dev.csv",
        job_status=status,
        started_at="2026-08-15T10:49:00+00:00",
        execution_backend=backend,
        stages=stages or [],
        error=error,
    )


def _service() -> DeploymentService:
    return DeploymentService.__new__(DeploymentService)


def test_a_pending_databricks_run_reports_compute_starting():
    """PENDING is Databricks acquiring compute -- starting a stopped
    existing cluster, or provisioning a new job cluster."""
    status = _service()._to_deployment_status(_listing())

    assert status.compute is not None
    assert status.compute.state == "starting"
    assert status.compute.label == "Starting Compute"
    assert "Starting the selected Databricks compute" in status.compute.message
    assert "few minutes" in status.compute.detail


def test_a_running_run_with_no_reported_stage_reports_compute_ready():
    """The PENDING -> RUNNING transition is the observed moment compute
    became ready: the task is executing, the engine has not reported yet."""
    status = _service()._to_deployment_status(_listing(status=JobStatus.RUNNING))

    assert status.compute is not None
    assert status.compute.state == "ready"
    assert status.compute.label == "Compute Ready"
    assert "Starting the forecast pipeline" in status.compute.message


def test_compute_status_disappears_once_the_engine_reports_a_stage():
    """From the first stage on, the phases say it better."""
    reported = [{"name": "Load Dataset", "status": "Running", "started_at": "t0"}]

    status = _service()._to_deployment_status(_listing(reported, status=JobStatus.RUNNING))

    assert status.compute is None
    assert status.current_stage == "Load & Prepare"


def test_local_runs_have_no_compute_to_start():
    status = _service()._to_deployment_status(_listing(backend="local"))

    assert status.compute is None


def test_a_run_that_failed_before_any_stage_reports_compute_unavailable():
    """A cluster that could not start (quota, policy) would otherwise show
    seven Pending phases and no explanation at all."""
    status = _service()._to_deployment_status(
        _listing(status=JobStatus.FAILED, error="Insufficient vCPU quota.")
    )

    assert status.compute is not None
    assert status.compute.state == "failed"
    assert status.compute.detail == "Insufficient vCPU quota."


def test_a_completed_run_reports_no_compute_status():
    status = _service()._to_deployment_status(
        _listing([{"name": name, "status": "Completed"} for name in ("Load Dataset",)],
                 status=JobStatus.COMPLETED)
    )

    assert status.compute is None


def test_compute_startup_never_becomes_an_eighth_forecast_phase():
    """The seven phases are the engine's own stages. Compute startup is
    infrastructure and is reported beside the trail, never inside it."""
    for status in (JobStatus.PENDING, JobStatus.RUNNING):
        deployment = _service()._to_deployment_status(_listing(status=status))

        assert [stage.label for stage in deployment.stages] == PHASE_LABELS
        assert all(stage.status == "Pending" for stage in deployment.stages)
        # And the denominator is untouched: no phase was completed.
        assert deployment.progress == 0


def test_a_compute_failure_explains_itself_with_no_trail_to_show():
    """A run that never started has no stage trail -- inventing seven
    Pending phases for it would be a fabrication. The compute status is
    then the only thing there is to say, so it must be there to say it."""
    deployment = _service()._to_deployment_status(
        _listing(status=JobStatus.FAILED, error="Insufficient vCPU quota.")
    )

    assert deployment.stages == []
    assert deployment.compute is not None
    assert deployment.compute.state == "failed"


def test_the_pending_run_names_compute_rather_than_queued():
    """"Queued" is what made a booting cluster read as a stalled app."""
    assert _service()._to_deployment_status(_listing()).current_stage == "Starting Compute"
