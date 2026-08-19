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

    # The whole pipeline is described, not just the part reached so far.
    assert [stage.label for stage in stages] == PIPELINE_STAGES

    by_label = {stage.label: stage for stage in stages}
    assert by_label["Load Dataset"].status == "Completed"
    assert by_label["Load Dataset"].completed_at == "t1"
    assert by_label["Assess Quality"].status == "Running"
    # Everything past the running stage is explicitly Pending.
    assert by_label["Train Models"].status == "Pending"
    assert by_label["MLflow Tracking"].status == "Pending"


def test_progress_is_measured_against_the_whole_pipeline():
    reported = [
        {"name": name, "status": "Completed"} for name in PIPELINE_STAGES[:6]
    ]

    status = _service()._to_deployment_status(_listing(reported))

    # 6 of 17 completed — not 6 of the 6 reported, and not 6 of a
    # short-by-three canonical list.
    assert status.progress == round(6 / len(PIPELINE_STAGES) * 100)
    assert status.current_stage == PIPELINE_STAGES[5]


def test_a_finished_run_with_no_trail_reports_no_stages():
    """Seventeen Pending stages for a run that already finished would be a
    fabrication, not a rendering of its shape."""
    assert _service()._to_stage_statuses(_listing([], status=JobStatus.COMPLETED)) == []


def test_every_databricks_task_key_titlecases_to_a_real_stage():
    """The naming contract: the Serverless job's task_keys and the stage
    trail are one vocabulary, so a task renamed in the YAML without its
    stage being renamed too is caught here rather than in a demo."""
    yaml_source = Path("databricks/resources/forecast_job_serverless.yml").read_text()
    # Task definitions sit at eight spaces; the deeper-indented `task_key`s
    # under a `depends_on:` are references to them, not new tasks.
    task_keys = re.findall(r"^ {8}- task_key: (\w+)$", yaml_source, re.MULTILINE)

    assert len(task_keys) == 11, task_keys

    # load_prepare and build_series fan out to several stages; the other
    # nine are exactly one stage each, named by Title-Casing the key.
    fan_out = {"load_prepare", "build_series"}
    # Two keys whose stage spelling is not pure Title Case, both for
    # readability in the UI rather than any behavioural reason.
    spelling = {"rank_select": "Rank & Select", "mlflow_tracking": "MLflow Tracking"}
    for key in task_keys:
        if key in fan_out:
            continue
        expected = spelling.get(key) or key.replace("_", " ").title()
        assert expected in PIPELINE_STAGES, f"{key} -> {expected!r} is not a stage"


def test_stage_names_stay_short_enough_to_read_in_the_trail():
    """The whole point of the rename: seventeen labels the UI can show
    without truncating. 'Rank & Select' is the only three-token label and
    its middle token is one character."""
    for label in PIPELINE_STAGES:
        assert len(label) <= 18, f"{label!r} is too long for the stage trail"


def test_a_run_recorded_under_the_old_stage_names_still_renders():
    """Runs completed before the vocabulary was unified are still on disk.
    Their trails must map onto the current skeleton, not pile up as
    unknown extras below seventeen Pending stages."""
    reported = [
        {"name": "Load Dataset", "status": "Completed"},
        {"name": "Assess Data Quality", "status": "Completed"},
        {"name": "Generate Explainability (SHAP)", "status": "Completed"},
        {"name": "Track to MLflow", "status": "Completed"},
    ]

    stages = _service()._to_stage_statuses(_listing(reported, status=JobStatus.COMPLETED))

    assert [stage.label for stage in stages] == PIPELINE_STAGES
    by_label = {stage.label: stage for stage in stages}
    assert by_label["Assess Quality"].status == "Completed"
    assert by_label["Explain Models"].status == "Completed"
    assert by_label["MLflow Tracking"].status == "Completed"


def test_an_unrecognised_reported_stage_is_kept_not_dropped():
    reported = [
        {"name": "Load Dataset", "status": "Completed"},
        {"name": "Some New Stage", "status": "Completed"},
    ]

    labels = [stage.label for stage in _service()._to_stage_statuses(_listing(reported))]

    assert labels[: len(PIPELINE_STAGES)] == PIPELINE_STAGES
    assert "Some New Stage" in labels
