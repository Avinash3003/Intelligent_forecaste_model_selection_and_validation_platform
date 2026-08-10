import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import (
    auth,
    deployment,
    estimation,
    execution,
    health,
    metadata,
    mlflow_view,
    profile,
    results,
    upload,
)
from app.auth.entra import AuthConfigurationError
from app.auth.dependencies import get_token_validator
from app.config.settings import get_settings
from app.utils.errors import safe_detail

logger = logging.getLogger("forecastiq.api")

settings = get_settings()

# Fail at import time, not per request. A deployment that is neither local
# development nor authenticated is a configuration mistake with a security
# consequence, and refusing to start is the only response that cannot be
# ignored.
if not settings.auth_enabled and settings.is_production_like:
    raise AuthConfigurationError(
        f"AUTH_ENABLED is false but APP_ENV is '{settings.app_env}'. "
        "Entra ID authentication is mandatory outside local development."
    )

if settings.auth_enabled:
    # Surfaces a missing tenant/audience now rather than as a blanket 401
    # on every request once the frontend is already deployed.
    get_token_validator().require_configuration()

app = FastAPI(
    title=settings.app_name,
    description="API for the Intelligent Forecast Model Selection & Validation Platform.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Report *what* is wrong with a request, never how it is parsed.

    FastAPI's default body enumerates pydantic locations and internal type
    names. Users get the field names and nothing else; the full detail is
    logged for whoever is debugging it.
    """
    logger.warning("Request validation failed for %s: %s", request.url.path, exc.errors())
    fields = sorted({str(error["loc"][-1]) for error in exc.errors() if error.get("loc")})
    detail = (
        f"Please check these fields and try again: {', '.join(fields)}."
        if fields
        else "The request was invalid. Please check your input and try again."
    )
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defence: no traceback, credential or endpoint escapes.

    The real exception is logged with its stack trace; the client receives
    a translated, redacted message (see `app/utils/errors.py`).
    """
    logger.exception("Unhandled error serving %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": safe_detail(exc)})


# Routed at the root today (matching the current frontend contract) rather
# than under settings.api_v1_prefix; switch to the prefix once the frontend
# is updated to call versioned paths.
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(profile.router)
app.include_router(metadata.router)
app.include_router(estimation.router)
app.include_router(deployment.router)
app.include_router(execution.router)
app.include_router(results.router)
app.include_router(mlflow_view.router)
