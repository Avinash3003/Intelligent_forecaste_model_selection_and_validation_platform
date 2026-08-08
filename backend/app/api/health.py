from fastapi import APIRouter

from app.config.settings import get_settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Service health check")
def get_health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", service=settings.app_name, version="0.1.0", environment=settings.app_env)
