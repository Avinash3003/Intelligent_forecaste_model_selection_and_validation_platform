"""The dataset inspection summary shown right after upload.

Same resolve -> load -> analyze structure as api/metadata.py, reusing
UploadService and DatasetLoader rather than duplicating file handling.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import require
from app.auth.models import Permission, Principal
from app.schemas.profile import DateRangeRequest, DateRangeResponse, ProfileRequest, ProfileResponse
from app.services.dataset_loader import DatasetLoader
from app.services.profile_service import ProfileService, compute_date_range
from app.services.upload_service import UploadService
from app.utils.exceptions import DatasetLoadError, FileResolutionError

router = APIRouter(prefix="/profile", tags=["Profile"])

upload_service = UploadService()
dataset_loader = DatasetLoader()
profile_service = ProfileService()


@router.post("", response_model=ProfileResponse, summary="Profile an uploaded dataset")
def profile_dataset(
    request: ProfileRequest,
    principal: Principal = Depends(require(Permission.DATASET_READ)),
) -> ProfileResponse:
    """Generate a basic schema/profile summary for a previously uploaded
    dataset. Called automatically by the frontend right after upload."""
    try:
        file_path, dataset_name = upload_service.resolve(request.file_id)
    except FileResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        dataframe = dataset_loader.load(file_path)
    except DatasetLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return profile_service.profile(dataframe, dataset_name, file_path.stat().st_size)


@router.post("/date-range", response_model=DateRangeResponse, summary="Observed date coverage for a candidate date column")
def get_date_range(
    request: DateRangeRequest,
    principal: Principal = Depends(require(Permission.DATASET_READ)),
) -> DateRangeResponse:
    """The dataset's real date coverage for the column being considered.

    Called as the selection changes, ahead of full validation. Uses the same
    parse-then-min/max rule the engine's quality stage does, duplicated
    rather than imported since the two are separate processes.
    """
    try:
        file_path, _dataset_name = upload_service.resolve(request.file_id)
    except FileResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        dataframe = dataset_loader.load(file_path)
    except DatasetLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.date_column not in dataframe.columns:
        raise HTTPException(status_code=400, detail=f"Column '{request.date_column}' does not exist in this dataset.")

    start, end = compute_date_range(dataframe[request.date_column])
    return DateRangeResponse(available=start is not None and end is not None, date_range_start=start, date_range_end=end)
