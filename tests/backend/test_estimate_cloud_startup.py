"""A cloud estimate must account for one compute cold start, not eleven.

The pipeline now submits exactly one task per run (jobs.submit), on either
an existing cluster starting on demand or a freshly provisioned job
cluster — never the old eleven-task Serverless DAG, where each task paid
its own startup. Charging that per-task figure here overstated a
single-task run's wait several times over.
"""

from __future__ import annotations

from app.services import estimation_service as es


def test_the_fallback_reflects_one_cold_start_not_several():
    """No task-count multiplication left: the constant is a single figure,
    not `per_task * task_count`."""
    assert es._CLOUD_STARTUP_SECONDS == 396.0


def test_the_fallback_lands_in_the_range_real_cold_starts_showed():
    """Sized from two real cold starts observed on this project's own
    workspace: 351s (a TERMINATED cluster starting on demand) and 441s (a
    job cluster provisioning from nothing)."""
    assert 351.0 <= es._CLOUD_STARTUP_SECONDS <= 441.0


def test_local_execution_is_charged_no_startup_at_all():
    service = es.EstimationService.__new__(es.EstimationService)
    assert service._startup_seconds("local", None) == 0.0


def test_a_measured_figure_always_wins_over_the_fallback():
    """History measures wall-clock minus engine time, which already covers
    the real startup — so once it exists it must override the constant."""
    service = es.EstimationService.__new__(es.EstimationService)
    assert service._startup_seconds("databricks", 512.0) == 512.0


def test_the_fallback_is_used_only_when_nothing_was_measured():
    service = es.EstimationService.__new__(es.EstimationService)
    assert service._startup_seconds("databricks", None) == es._CLOUD_STARTUP_SECONDS
