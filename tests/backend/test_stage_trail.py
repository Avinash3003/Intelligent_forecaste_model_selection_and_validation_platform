"""The Deployments stage trail always describes the whole pipeline.

The engine reports a stage only once it has begun it, so a live run's trail
grows one entry at a time. Returning that trail verbatim made the UI drop
every stage not yet reached: a run partway through Preprocess rendered as a
six-stage pipeline that simply ended there, with no indication that eleven
more were still to come, and the progress percentage was computed against a
canonical list that was itself missing three real stages.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.orchestration.schemas import JobStatus, RunListing
from app.services.deployment_service import PIPELINE_STAGES, DeploymentService
from app.services.pipeline_stages import PHASE_LABELS


def _listing(stages: list[dict], status: JobStatus = JobStatus.RUNNING) -> RunListing:
    return RunListing(
        run_id="dbx-run-test",
        dataset_name="store_item_dev.csv",
        job_status=status,
        started_at="2026-08-15T10:49:00+00:00",
        execution_backend="databricks",
        stages=stages,
    )


def _service() -> DeploymentService:
    return DeploymentService.__new__(DeploymentService)


def test_canonical_stage_list_matches_the_engine_exactly():
    """PIPELINE_STAGES is the progress denominator, so a stage missing from
    it silently overstates how far along every run is."""
    source = Path("forecast_engine/run_pipeline.py").read_text()
    engine_stages = re.findall(r'begin_stage\("([^"]+)"\)', source)

    assert engine_stages == PIPELINE_STAGES


def test_unreached_stages_are_reported_as_pending_not_omitted():
    reported = [
        {"name": "Load Dataset", "status": "Completed", "started_at": "t0", "completed_at": "t1"},
        {"name": "Detect Frequency", "status": "Completed", "started_at": "t1", "completed_at": "t2"},
        {"name": "Assess Quality", "status": "Running", "started_at": "t2"},
    ]

    stages = _service()._to_stage_statuses(_listing(reported))

    # The whole pipeline is described as seven phases, not just the part
    # reached so far.
    assert [stage.label for stage in stages] == PHASE_LABELS

    by_label = {stage.label: stage for stage in stages}
    # Load & Prepare owns six stages; three are reported and one is still
    # Running, so the phase is Running — never Completed on a partial.
    assert by_label["Load & Prepare"].status == "Running"
    assert by_label["Load & Prepare"].started_at == "t0"
    assert by_label["Load & Prepare"].completed_at is None
    # Everything past it is explicitly Pending.
    assert by_label["Train Models"].status == "Pending"
    assert by_label["Publish Results"].status == "Pending"


def test_progress_is_measured_against_the_whole_pipeline():
    reported = [
        {"name": name, "status": "Completed"} for name in PIPELINE_STAGES[:6]
    ]

    status = _service()._to_deployment_status(_listing(reported))

    # Progress still counts the ENGINE's seventeen stages, so it moves
    # smoothly rather than in jumps of a seventh: 6 of 17, not 1 of 7.
    assert status.progress == round(6 / len(PIPELINE_STAGES) * 100)
    # The trail itself reads in phases, so the current step is a phase name.
    assert status.current_stage == "Load & Prepare"


def test_a_finished_run_with_no_trail_reports_no_stages():
    """Seventeen Pending stages for a run that already finished would be a
    fabrication, not a rendering of its shape."""
    assert _service()._to_stage_statuses(_listing([], status=JobStatus.COMPLETED)) == []


def test_stage_names_stay_short_enough_to_read_in_the_trail():
    """The whole point of the rename: seventeen labels the UI can show
    without truncating. 'Rank & Select' is the only three-token label and
    its middle token is one character."""
    for label in PIPELINE_STAGES:
        assert len(label) <= 18, f"{label!r} is too long for the stage trail"


def test_a_run_recorded_under_the_old_stage_names_still_renders():
    """Runs completed before the vocabulary was unified are still on disk.
    Their trails must map onto the current phases, not pile up as unknown
    extras below seven Pending ones."""
    reported = [
        {"name": "Load Dataset", "status": "Completed"},
        {"name": "Assess Data Quality", "status": "Completed"},
        {"name": "Generate Explainability (SHAP)", "status": "Completed"},
        {"name": "Track to MLflow", "status": "Completed"},
    ]

    stages = _service()._to_stage_statuses(_listing(reported, status=JobStatus.COMPLETED))

    assert [stage.label for stage in stages] == PHASE_LABELS
    by_label = {stage.label: stage for stage in stages}
    # The legacy names resolved into their phases rather than appearing as
    # unknown extras: Explain Models is a whole phase and reported complete,
    # while Load & Prepare saw only 2 of its 6 stages so it is not.
    assert by_label["Explain Models"].status == "Completed"
    assert by_label["Load & Prepare"].status == "Running"
    # No legacy name leaked through as its own row.
    assert "Assess Data Quality" not in by_label
    assert "Generate Explainability (SHAP)" not in by_label
    assert "Track to MLflow" not in by_label


def test_an_unrecognised_reported_stage_is_kept_not_dropped():
    reported = [
        {"name": "Load Dataset", "status": "Completed"},
        {"name": "Some New Stage", "status": "Completed"},
    ]

    labels = [stage.label for stage in _service()._to_stage_statuses(_listing(reported))]

    assert labels[: len(PHASE_LABELS)] == PHASE_LABELS
    assert "Some New Stage" in labels
