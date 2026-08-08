from pydantic import BaseModel

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
