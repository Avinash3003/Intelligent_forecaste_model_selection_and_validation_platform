"""Profile API — generates the Basic Dataset Inspection summary shown right
after upload, before any metadata mapping happens (Section 5.1.1).

Mirrors the same resolve -> load -> analyze structure as api/metadata.py,
reusing UploadService and DatasetLoader rather than duplicating file
resolution/parsing logic.
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
    """The dataset's real date coverage for whichever column the user has
    tentatively assigned as the date column in Metadata Mapping (Priority
    B) — called as that selection changes, ahead of full "/metadata/validate".

    Mirrors the same parse-then-min/max rule a real run's Assess Data
    Quality stage uses (`forecast_engine.s02_quality.quality_assessor.
    parse_date_column` / `date_range_from_parsed`) — deliberately not a
    cross-process import of it: the backend and forecast_engine are two
    separate deployable processes with separate dependencies (the backend
    invokes forecast_engine only as an external subprocess/job, never
    in-process — see LocalRunner/DatabricksRunner), so this stays a small,
    self-contained duplicate of that rule rather than a new coupling.
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
