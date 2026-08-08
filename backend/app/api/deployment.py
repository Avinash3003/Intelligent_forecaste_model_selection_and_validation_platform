from fastapi import APIRouter, HTTPException

from app.orchestration.exceptions import ExecutionError, UnknownRunError
from app.schemas.deployment import DeploymentRequest, DeploymentResponse, DeploymentStatus
from app.services.deployment_service import DeploymentService
from app.utils.exceptions import FileResolutionError

router = APIRouter(tags=["Deployment"])
deployment_service = DeploymentService()


@router.post("/deploy", response_model=DeploymentResponse, summary="Submit a forecasting run")
def deploy(request: DeploymentRequest) -> DeploymentResponse:
    if not request.file_id:
        raise HTTPException(status_code=400, detail="file_id is required to submit a forecasting run.")

    try:
        return deployment_service.deploy(request)
    except FileResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/deployments", response_model=list[DeploymentStatus], summary="List deployment history")
def list_deployments() -> list[DeploymentStatus]:
    return deployment_service.list_deployments()


@router.get(
    "/deployments/{run_id}",
    response_model=DeploymentStatus,
    summary="Status detail for one run",
)
def get_deployment(run_id: str) -> DeploymentStatus:
    try:
        return deployment_service.get_deployment(run_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
