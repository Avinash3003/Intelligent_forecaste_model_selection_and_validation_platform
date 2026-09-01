"""Runs Dataset Loader -> Metadata Interpreter -> Validation Engine.

The only place that resolves a file_id to a file; every service it calls
takes a Path or a DataFrame, which keeps them independently testable.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.auth.dependencies import require
from app.auth.models import Permission, Principal
from app.config.model_availability import CANDIDATE_MODEL_IDS, DEFAULT_FALLBACK_MODEL, unavailable_models
from app.config.run_limits import DEFAULT_FORECAST_HORIZON, MAX_FORECAST_HORIZON, MIN_FORECAST_HORIZON
from app.config.settings import Settings, get_settings
from app.schemas.metadata import (
    ForecastHorizonRange,
    MetadataRequest,
    MetadataValidationResponse,
    ModelAvailability,
    ModelAvailabilityResponse,
)
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
    """Validate the user's column-role selections against their dataset."""
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


@router.get("/models", response_model=ModelAvailabilityResponse, summary="Candidate models runnable on this execution mode")
def get_model_availability(
    settings: Settings = Depends(get_settings),
    principal: Principal = Depends(require(Permission.FORECAST_CONFIGURE)),
) -> ModelAvailabilityResponse:
    """Which candidate models the configured execution mode can actually run.

    Read by the Configure step so it never offers a model whose library is
    absent from the environment the run will execute in.
    """
    mode = (settings.execution_mode or "local").strip().lower()
    blocked = unavailable_models(mode)
    return ModelAvailabilityResponse(
        execution_mode=mode,
        models=[
            ModelAvailability(
                id=model_id,
                available=model_id not in blocked,
                reason=blocked.get(model_id),
            )
            for model_id in CANDIDATE_MODEL_IDS
        ],
        horizon=ForecastHorizonRange(
            min_months=MIN_FORECAST_HORIZON,
            max_months=MAX_FORECAST_HORIZON,
            default_months=DEFAULT_FORECAST_HORIZON,
        ),
        default_fallback_model=DEFAULT_FALLBACK_MODEL,
    )
