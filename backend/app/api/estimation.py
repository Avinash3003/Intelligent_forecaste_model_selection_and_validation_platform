"""The Estimate step between Configure and Run.

Reuses the same resolve -> load path as /profile, and runs no forecasting
code: the estimate comes from the dataset's shape and the model mix.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import require
from app.auth.models import Permission, Principal
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.services.dataset_loader import DatasetLoader
from app.services.estimation_service import EstimationService, get_estimation_service
from app.services.upload_service import UploadService
from app.utils.exceptions import DatasetLoadError, FileResolutionError

router = APIRouter(prefix="/estimate", tags=["Estimation"])

upload_service = UploadService()
dataset_loader = DatasetLoader()


@router.post("", response_model=EstimationResponse, summary="Estimate a run's duration and compute cost")
def estimate_run(
    request: EstimationRequest,
    service: EstimationService = Depends(get_estimation_service),
    principal: Principal = Depends(require(Permission.FORECAST_ESTIMATE)),
) -> EstimationResponse:
    try:
        file_path, _ = upload_service.resolve(request.file_id)
    except FileResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        dataframe = dataset_loader.load(file_path)
    except DatasetLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return service.estimate(dataframe, request)
