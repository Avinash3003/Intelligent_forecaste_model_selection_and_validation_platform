from fastapi import APIRouter, Depends, UploadFile

from app.auth.dependencies import require
from app.auth.models import Permission, Principal
from app.schemas.upload import UploadResponse
from app.services.upload_service import UploadService

router = APIRouter(prefix="/upload", tags=["Upload"])
upload_service = UploadService()


@router.post("", response_model=UploadResponse, summary="Upload a time-series dataset")
def upload_dataset(
    file: UploadFile,
    principal: Principal = Depends(require(Permission.DATASET_UPLOAD)),
) -> UploadResponse:
    # Datasets are staged server-side and handed to the pipeline from there;
    # the browser never uploads to Databricks or to storage directly, so no
    # storage credential is ever exposed to it.
    return upload_service.save(file)
