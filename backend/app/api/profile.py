"""Profile API — generates the Basic Dataset Inspection summary shown right
after upload, before any metadata mapping happens (Section 5.1.1).

Mirrors the same resolve -> load -> analyze structure as api/metadata.py,
reusing UploadService and DatasetLoader rather than duplicating file
resolution/parsing logic.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.profile import ProfileRequest, ProfileResponse
from app.services.dataset_loader import DatasetLoader
from app.services.profile_service import ProfileService
from app.services.upload_service import UploadService
from app.utils.exceptions import DatasetLoadError, FileResolutionError

router = APIRouter(prefix="/profile", tags=["Profile"])

upload_service = UploadService()
dataset_loader = DatasetLoader()
profile_service = ProfileService()


@router.post("", response_model=ProfileResponse, summary="Profile an uploaded dataset")
def profile_dataset(request: ProfileRequest) -> ProfileResponse:
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
