from pydantic import BaseModel, Field

from app.orchestration.schemas import ExecutionBackend, JobStatus


class ExecutionSubmitResponse(BaseModel):
    """Response for `POST /execution/submit` — intentionally minimal.

    The full `PipelineExecutionResult` is only meaningful once a run
    reaches a terminal status; immediately after submission there is
    nothing more to report than "here is your run id, and here is where it
    stands right now".
    """

    run_id: str
    job_status: JobStatus
    execution_backend: ExecutionBackend


class ExecutionStatusResponse(BaseModel):
    run_id: str
    job_status: JobStatus


class ExecutionCancelResponse(BaseModel):
    run_id: str
    cancelled: bool
    # Which run-scoped storage location(s), if any, could not be cleaned up
    # — empty means every one succeeded (or was already empty). Never a
    # bare bool: a cancellation can stop the run while only some of its
    # storage locations fail to clean up, and a caller needs to know which.
    cleanup_errors: list[str] = Field(default_factory=list)
