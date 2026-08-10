"""Metadata API — orchestrates Dataset Loader -> Metadata Interpreter ->
Validation Engine (Sections 5.1.2 / 6.2 of the design doc).

This route is the only place that knows how to resolve a `file_id` into an
actual file; every service it calls works with either a Path or an
already-loaded DataFrame, which keeps them independently testable and easy
to swap out (e.g. UploadService -> Azure Blob Storage) later.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import require
from app.auth.models import Permission, Principal
from app.schemas.metadata import MetadataRequest, MetadataValidationResponse
from app.services.dataset_loader import DatasetLoader
from app.services.metadata_interpreter import MetadataInterpreter
from app.services.upload_service import UploadService
from app.services.validation_engine import ValidationEngine
from app.utils.exceptions import DatasetLoadError, FileResolutionError

router = APIRouter(prefix="/metadata", tags=["Metadata"])

upload_service = UploadService()
dataset_loader = DatasetLoader()
metadata_interpreter = MetadataInterpreter()
validation_engine = ValidationEngine()


@router.post("/validate", response_model=MetadataValidationResponse, summary="Interpret and validate metadata mapping")
def validate_metadata(
    request: MetadataRequest,
    principal: Principal = Depends(require(Permission.FORECAST_CONFIGURE)),
) -> MetadataValidationResponse:
    """Validate a user's Date/Target/Key/Feature column selections against
    the dataset they previously uploaded.

    Called by the frontend's Metadata Mapping step once the user has
    chosen every required column role.
    """
    # Resolve file_id -> path first; every later step needs real data to
    # validate against, so there's nothing useful to do without it.
    try:
        file_path, dataset_name = upload_service.resolve(request.file_id)
    except FileResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        dataframe = dataset_loader.load(file_path)
    except DatasetLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized_config = metadata_interpreter.interpret(dataframe, request)
    return validation_engine.validate(dataframe, normalized_config, dataset_name)
