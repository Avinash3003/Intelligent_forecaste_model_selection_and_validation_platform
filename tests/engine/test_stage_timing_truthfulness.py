"""A stage that did work must not report 0 seconds.

Ray runs train -> evaluate -> explain -> rank -> select for one key inside a
single task. By the time the driver opens its "Evaluate Models" stage the
work has already finished, so driver elapsed time reads ~0s for four real
stages and the whole parallel window lands on "Train Models".

Observed on real runs (dbx-run-26be018dc3f6, dbx-run-ed751d39db78):

    Evaluate Models  Completed  0.0s
      detail: 1 survived, 2 eliminated, 0 failed across 3 group(s).
              15 model fit(s)
    evaluation_report.duration_seconds: 1.664
      backtest 1.493s, forecast_generation 0.154s, validation 0.017s

The work was real. Only the measurement was wrong, and a 0s "Completed"
reads to a user exactly like a stage that was skipped.
"""

from __future__ import annotations

import time

from forecast_engine.core.pipeline_context import PipelineContext, StageRecord


def _context():
    return PipelineContext.create(
        dataset_path="x.csv", configuration=None, pipeline_config=None, run_id="r1"
    )


def test_a_stage_reports_the_duration_it_measured_itself():
    context = _context()
    record = context.begin_stage("Evaluate Models")

    context.complete_stage(record, detail="done", measured_seconds=1.664)

    assert record.duration_seconds == 1.664


def test_the_driver_clock_is_kept_but_never_confused_with_the_real_work():
    """Both numbers survive: one is what the stage did, the other is how
    long the driver spent orchestrating it."""
    context = _context()
    record = context.begin_stage("Evaluate Models")

    context.complete_stage(record, detail="done", measured_seconds=1.664)
    payload = record.to_dict()

    assert payload["duration_seconds"] == 1.664
    assert payload["measured_seconds"] == 1.664
    assert payload["orchestration_seconds"] < 1.0  # the driver really was ~0s


def test_a_stage_without_its_own_measurement_still_uses_the_wall_clock():
    """Sequential stages are unaffected -- nothing regresses for the stages
    that genuinely run on the driver."""
    context = _context()
    record = context.begin_stage("Load Dataset")
    time.sleep(0.02)

    context.complete_stage(record, detail="loaded")

    assert record.measured_seconds is None
    assert record.duration_seconds >= 0.02


def test_a_completed_stage_is_never_marked_skipped():
    """The rule this file defends: 0s is not evidence of a skip."""
    context = _context()
    record = context.begin_stage("Explain Models")

    context.complete_stage(record, detail="Explainability generated for 1 model(s).")

    assert record.status == "Completed"
    assert record.status != "Skipped"


def test_the_running_stage_reports_no_duration_yet():
    record = StageRecord(name="Train Models")

    assert record.duration_seconds is None
    assert record.to_dict()["measured_seconds"] is None
