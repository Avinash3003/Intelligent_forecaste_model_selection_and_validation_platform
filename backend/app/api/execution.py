"""Direct access to the Pipeline Executor (Section 6.14).

Exposes the orchestration layer's own standardized objects
(`JobStatus`, `PipelineExecutionResult`) rather than the legacy
dashboard-shaped `/deploy` + `/results/{run_id}` contract those routes
still serve — this is the "communicates only with the Pipeline Executor"
surface Section 6.14 asks for, kept as an additive API so the existing
frontend contract is undisturbed.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config.settings import get_settings
from app.orchestration.exceptions import ExecutionError, RunNotReadyError, UnknownRunError
from app.orchestration.executor import get_pipeline_executor
from app.orchestration.schemas import ExecutionBackend, PipelineExecutionResult
from app.schemas.deployment import DeploymentRequest
from app.schemas.execution import ExecutionCancelResponse, ExecutionStatusResponse, ExecutionSubmitResponse
from app.services.deployment_service import build_execution_request
from app.services.upload_service import UploadService

router = APIRouter(prefix="/execution", tags=["Execution"])
upload_service = UploadService()


@router.post("/submit", response_model=ExecutionSubmitResponse, summary="Submit a run through the Pipeline Executor")
def submit_execution(request: DeploymentRequest) -> ExecutionSubmitResponse:
    if not request.file_id:
        raise HTTPException(status_code=400, detail="file_id is required to submit a pipeline execution.")

    try:
        dataset_path, original_filename = upload_service.resolve(request.file_id)
        execution_request = build_execution_request(request, dataset_path)
        execution_request.dataset_name = execution_request.dataset_name or original_filename
        executor = get_pipeline_executor()
        run_id = executor.execute(execution_request)
        status = executor.get_status(run_id)
    except ExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ExecutionSubmitResponse(
        run_id=run_id,
        job_status=status,
        execution_backend=ExecutionBackend(get_settings().execution_mode),
    )


@router.get("/{run_id}/status", response_model=ExecutionStatusResponse, summary="Poll a run's execution status")
def get_execution_status(run_id: str) -> ExecutionStatusResponse:
    try:
        status = get_pipeline_executor().get_status(run_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ExecutionStatusResponse(run_id=run_id, job_status=status)


@router.get(
    "/{run_id}/result",
    response_model=PipelineExecutionResult,
    summary="Retrieve the standardized PipelineExecutionResult for a completed run",
)
def get_execution_result(run_id: str) -> PipelineExecutionResult:
    try:
        return get_pipeline_executor().get_result(run_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RunNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{run_id}/cancel", response_model=ExecutionCancelResponse, summary="Cancel a pending or running job")
def cancel_execution(run_id: str) -> ExecutionCancelResponse:
    try:
        cancelled = get_pipeline_executor().cancel(run_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ExecutionCancelResponse(run_id=run_id, cancelled=cancelled)
