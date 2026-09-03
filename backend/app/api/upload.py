from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.auth.dependencies import require
from app.auth.models import Permission, Principal
from app.schemas.upload import UploadResponse
from app.services.upload_service import UploadService
from app.utils.exceptions import UploadTooLargeError

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
    try:
        return upload_service.save(file)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
