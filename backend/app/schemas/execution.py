from pydantic import BaseModel, Field

from app.orchestration.schemas import ExecutionBackend, JobStatus


class ExecutionSubmitResponse(BaseModel):
    """Deliberately minimal: right after submission there is nothing to report
    beyond the run id and its current status."""

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
