from fastapi import APIRouter, UploadFile

from app.schemas.upload import UploadResponse
from app.services.upload_service import UploadService

router = APIRouter(prefix="/upload", tags=["Upload"])
upload_service = UploadService()


@router.post("", response_model=UploadResponse, summary="Upload a time-series dataset")
def upload_dataset(file: UploadFile) -> UploadResponse:
    return upload_service.save(file)
