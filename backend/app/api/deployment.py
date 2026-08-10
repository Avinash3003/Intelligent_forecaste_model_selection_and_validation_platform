from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import require
from app.auth.models import Permission, Principal
from app.orchestration.exceptions import ExecutionError, UnknownRunError
from app.schemas.deployment import DeploymentRequest, DeploymentResponse, DeploymentStatus
from app.services.deployment_service import DeploymentService
from app.utils.errors import safe_detail
from app.utils.exceptions import FileResolutionError

router = APIRouter(tags=["Deployment"])
deployment_service = DeploymentService()


@router.post("/deploy", response_model=DeploymentResponse, summary="Submit a forecasting run")
def deploy(
    request: DeploymentRequest,
    principal: Principal = Depends(require(Permission.FORECAST_RUN)),
) -> DeploymentResponse:
    if not request.file_id:
        raise HTTPException(status_code=400, detail="Please upload a dataset before submitting a run.")

    try:
        return deployment_service.deploy(request)
    except FileResolutionError as exc:
        raise HTTPException(
            status_code=404,
            detail="That dataset is no longer available. Please upload it again.",
        ) from exc
    except ExecutionError as exc:
        # Translated and redacted: a Runner failure can carry workspace
        # URLs and secret-resolution internals, none of which belongs in
        # an HTTP response.
        raise HTTPException(
            status_code=502,
            detail=safe_detail(exc, fallback="The forecast could not be started. Please try again."),
        ) from exc


@router.get("/deployments", response_model=list[DeploymentStatus], summary="List deployment history")
def list_deployments(
    principal: Principal = Depends(require(Permission.RUN_READ)),
) -> list[DeploymentStatus]:
    return deployment_service.list_deployments()


@router.get(
    "/deployments/{run_id}",
    response_model=DeploymentStatus,
    summary="Status detail for one run",
)
def get_deployment(
    run_id: str,
    principal: Principal = Depends(require(Permission.RUN_READ)),
) -> DeploymentStatus:
    try:
        return deployment_service.get_deployment(run_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail="That run could not be found.") from exc
