"""Direct access to the Pipeline Executor (Section 6.14).

Exposes the orchestration layer's own standardized objects
(`JobStatus`, `PipelineExecutionResult`) rather than the legacy
dashboard-shaped `/deploy` + `/results/{run_id}` contract those routes
still serve — this is the "communicates only with the Pipeline Executor"
surface Section 6.14 asks for, kept as an additive API so the existing
frontend contract is undisturbed.

Every route here is a thin translation between HTTP and the Executor. No
forecasting logic lives in this layer, and no route talks to Databricks,
storage or MLflow directly — the selected Runner owns all of that.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import require
from app.auth.models import Permission, Principal
from app.config.settings import get_settings
from app.orchestration.exceptions import ExecutionError, RunNotReadyError, UnknownRunError
from app.orchestration.executor import get_pipeline_executor
from app.orchestration.schemas import ExecutionBackend, PipelineExecutionResult
from app.schemas.deployment import DeploymentRequest
from app.schemas.execution import ExecutionCancelResponse, ExecutionStatusResponse, ExecutionSubmitResponse
from app.services.deployment_service import build_execution_request
from app.services.derived_feature_registry import validate_derived_features
from app.services.upload_service import UploadService
from app.utils.errors import safe_detail
from app.utils.exceptions import FileResolutionError

router = APIRouter(prefix="/execution", tags=["Execution"])
upload_service = UploadService()


@router.post("/submit", response_model=ExecutionSubmitResponse, summary="Submit a run through the Pipeline Executor")
def submit_execution(
    request: DeploymentRequest,
    principal: Principal = Depends(require(Permission.FORECAST_RUN)),
) -> ExecutionSubmitResponse:
    if not request.file_id:
        raise HTTPException(status_code=400, detail="Please upload a dataset before submitting a run.")

    if request.derived_features is not None:
        unsupported = validate_derived_features(request.derived_features)
        if unsupported:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported derived feature(s): {', '.join(unsupported)}.",
            )

    try:
        dataset_path, original_filename = upload_service.resolve(request.file_id)
        execution_request = build_execution_request(request, dataset_path, principal)
        execution_request.dataset_name = execution_request.dataset_name or original_filename
        executor = get_pipeline_executor()
        run_id = executor.execute(execution_request)
        status = executor.get_status(run_id)
    except FileResolutionError as exc:
        raise HTTPException(
            status_code=404,
            detail="That dataset is no longer available. Please upload it again.",
        ) from exc
    except ExecutionError as exc:
        raise HTTPException(
            status_code=502,
            detail=safe_detail(exc, fallback="The forecast could not be started. Please try again."),
        ) from exc

    return ExecutionSubmitResponse(
        run_id=run_id,
        job_status=status,
        execution_backend=ExecutionBackend(get_settings().execution_mode),
    )


@router.get("/{run_id}/status", response_model=ExecutionStatusResponse, summary="Poll a run's execution status")
def get_execution_status(
    run_id: str,
    principal: Principal = Depends(require(Permission.RUN_READ)),
) -> ExecutionStatusResponse:
    """The status the frontend polls while a run executes.

    This is what makes opening the Databricks workspace unnecessary: it
    reports QUEUED/RUNNING/SUCCESS/FAILED in the platform's own vocabulary
    for both execution backends, and `/deployments/{run_id}` carries the
    per-stage trail behind it.
    """
    try:
        status = get_pipeline_executor().get_status(run_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail="That run could not be found.") from exc
    except ExecutionError as exc:
        raise HTTPException(status_code=502, detail=safe_detail(exc)) from exc

    return ExecutionStatusResponse(run_id=run_id, job_status=status)


@router.get(
    "/{run_id}/result",
    response_model=PipelineExecutionResult,
    summary="Retrieve the standardized PipelineExecutionResult for a completed run",
)
def get_execution_result(
    run_id: str,
    principal: Principal = Depends(require(Permission.RESULTS_READ)),
) -> PipelineExecutionResult:
    try:
        return get_pipeline_executor().get_result(run_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail="That run could not be found.") from exc
    except RunNotReadyError as exc:
        raise HTTPException(status_code=409, detail="This run has not finished yet.") from exc
    except ExecutionError as exc:
        raise HTTPException(status_code=502, detail=safe_detail(exc)) from exc


@router.post("/{run_id}/cancel", response_model=ExecutionCancelResponse, summary="Cancel a pending or running job")
def cancel_execution(
    run_id: str,
    principal: Principal = Depends(require(Permission.RUN_CANCEL)),
) -> ExecutionCancelResponse:
    try:
        outcome = get_pipeline_executor().cancel(
            run_id,
            cancelled_by_user_id=principal.subject,
            cancelled_by_display_name=principal.display_name or principal.subject,
        )
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail="That run could not be found.") from exc
    except ExecutionError as exc:
        raise HTTPException(
            status_code=502,
            detail=safe_detail(exc, fallback="The run could not be cancelled. Please try again."),
        ) from exc

    return ExecutionCancelResponse(run_id=run_id, cancelled=outcome.cancelled, cleanup_errors=outcome.cleanup_errors)
