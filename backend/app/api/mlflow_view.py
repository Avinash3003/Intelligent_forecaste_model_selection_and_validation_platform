"""MLflow Experiments page routes."""

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import require
from app.auth.models import Permission, Principal
from app.orchestration.exceptions import RunNotReadyError, UnknownRunError
from app.schemas.mlflow_view import MLflowRunDetail
from app.services.mlflow_view_service import MLflowViewService, get_mlflow_view_service

router = APIRouter(tags=["mlflow"])


@router.get("/mlflow/runs/{run_id}", response_model=MLflowRunDetail)
def get_mlflow_run(
    run_id: str,
    service: MLflowViewService = Depends(get_mlflow_view_service),
    principal: Principal = Depends(require(Permission.MODEL_INSPECT)),
) -> MLflowRunDetail:
    try:
        return service.get_run(run_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RunNotReadyError as exc:
        # A run still executing has no tracking record to show yet; 409 keeps
        # that distinct from "this run does not exist".
        raise HTTPException(status_code=409, detail=str(exc)) from exc
