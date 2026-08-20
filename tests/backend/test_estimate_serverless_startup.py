"""A Serverless estimate must account for every task's startup, not one.

The pipeline is deployed as an 11-task DAG, and each task acquires its own
compute and reloads the checkpointed context. Charging a single startup made
a fresh deployment (no history to calibrate from) quote "1-3 min" for a run
that really took 6-7 — the engine's own stage time was ~54s of a ~7 min wall
clock, so almost all of the elapsed time was orchestration the estimate never
counted.
"""

from __future__ import annotations

from app.services import estimation_service as es


def test_the_fallback_models_every_task_in_the_dag():
    assert es._SERVERLESS_STARTUP_SECONDS == (
        es._SERVERLESS_TASK_STARTUP_SECONDS * es._SERVERLESS_TASK_COUNT
    )


def test_the_task_count_matches_the_deployed_dag():
    """If the DAG gains or loses a task, this constant has to move with it —
    otherwise the estimate silently drifts from the run again."""
    import pathlib

    import yaml

    spec = yaml.safe_load(
        pathlib.Path("databricks/resources/forecast_job_serverless.yml").read_text()
    )
    tasks = spec["resources"]["jobs"]["forecast_pipeline_serverless"]["tasks"]
    assert es._SERVERLESS_TASK_COUNT == len(tasks)


def test_the_fallback_lands_in_the_range_a_real_run_showed():
    """~6 min of orchestration, matching the observed 6-7 min run — not the
    ~45s a single-task assumption produced."""
    assert 300 <= es._SERVERLESS_STARTUP_SECONDS <= 420


def test_local_execution_is_charged_no_startup_at_all():
    service = es.EstimationService.__new__(es.EstimationService)
    assert service._startup_seconds("local", None) == 0.0


def test_a_measured_figure_always_wins_over_the_fallback():
    """History measures wall-clock minus engine time, which already covers
    every task — so once it exists it must override the constant."""
    service = es.EstimationService.__new__(es.EstimationService)
    assert service._startup_seconds("databricks", 512.0) == 512.0


def test_the_fallback_is_used_only_when_nothing_was_measured():
    service = es.EstimationService.__new__(es.EstimationService)
    assert service._startup_seconds("databricks", None) == es._SERVERLESS_STARTUP_SECONDS
