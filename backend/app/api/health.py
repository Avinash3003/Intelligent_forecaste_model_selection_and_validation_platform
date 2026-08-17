from fastapi import APIRouter, Depends

from app.config.settings import Settings, get_settings
from app.orchestration.mlflow_history import MLflowHistoryStore
from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Service health check")
def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    # A fresh store, not the shared singleton: this route must never hold a
    # connection open, and its only job is a point-in-time reachability
    # check. is_available() never raises — an unreachable store degrades
    # this field to False rather than 500ing the health check itself.
    history = MLflowHistoryStore(settings)
    tracking_uri = settings.mlflow_tracking_uri_resolved
    backend = "databricks" if tracking_uri.startswith("databricks") else tracking_uri.split(":", 1)[0]

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version="0.1.0",
        environment=settings.app_env,
        execution_backend=(settings.execution_mode or "local").strip().lower(),
        run_history_available=history.is_available(),
        run_history_backend=backend,
    )
