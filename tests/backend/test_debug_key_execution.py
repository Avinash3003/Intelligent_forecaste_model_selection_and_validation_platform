"""The Ray key-level telemetry has to reach the browser as a named field.

It always travelled as far as the debug endpoint — but only nested four
levels deep inside `raw_result`, whose stated purpose is being an
unmodified dump of the result envelope. A UI built against that path would
be coupled to the envelope's internal shape and would break silently if it
were ever restructured. These pin the named field instead.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.orchestration.schemas import ExecutionBackend, JobStatus, PipelineExecutionResult
from app.services.debug_service import DebugService

KEY_EXECUTION = {
    "executor": "ray",
    "ray_cpus": 4.0,
    "max_concurrent_keys": 4,
    "wall_seconds": 317.489,
    "key_spans": [
        {"group_id": "G1", "worker_id": "w-aaa", "node_id": "n-1", "start": 0.0, "end": 17.2},
    ],
}


def _service(execution_summary: dict) -> DebugService:
    result = PipelineExecutionResult(
        run_id="run-1",
        job_status=JobStatus.COMPLETED,
        execution_backend=ExecutionBackend.DATABRICKS,
        execution_summary=execution_summary,
    )
    service = DebugService.__new__(DebugService)
    service._executor = SimpleNamespace(get_result=lambda run_id: result)
    return service


def test_key_execution_is_exposed_as_its_own_field():
    summary = _service({"metadata": {"key_execution": KEY_EXECUTION}}).get_debug_summary("run-1")

    assert summary.key_execution == KEY_EXECUTION


def test_key_spans_survive_intact_for_the_parallel_view():
    """The span list is what the timeline draws -- it must arrive whole,
    not summarised or reshaped on the way out."""
    summary = _service({"metadata": {"key_execution": KEY_EXECUTION}}).get_debug_summary("run-1")

    spans = summary.key_execution["key_spans"]
    assert spans[0]["worker_id"] == "w-aaa"
    assert spans[0]["node_id"] == "n-1"
    assert spans[0]["start"] == 0.0
    assert spans[0]["end"] == 17.2


def test_a_sequential_run_reports_no_key_execution():
    """Sequential execution records no key_execution at all, and the field
    must read as absent rather than an empty object the UI would try to
    draw."""
    summary = _service({"metadata": {"models_trained": 12}}).get_debug_summary("run-1")

    assert summary.key_execution is None


def test_a_run_with_no_metadata_at_all_does_not_raise():
    """A run that failed before Train Models has no metadata block. That is
    a normal state for this endpoint, which deliberately accepts unfinished
    runs."""
    summary = _service({}).get_debug_summary("run-1")

    assert summary.key_execution is None
