import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import (
    auth,
    compute,
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
from app.orchestration.executor import get_pipeline_executor
from app.utils.errors import safe_detail

logger = logging.getLogger("forecastiq.api")

settings = get_settings()

# Refuse to start rather than serve an unauthenticated production deployment.
if not settings.auth_enabled and settings.is_production_like:
    raise AuthConfigurationError(
        f"AUTH_ENABLED is false but APP_ENV is '{settings.app_env}'. "
        "Entra ID authentication is mandatory outside local development."
    )

if settings.auth_enabled:
    # Catch a missing tenant/audience now, not as a 401 on every request later.
    get_token_validator().require_configuration()

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the slow reads before the first request has to pay for them.

    Reading run history from a Databricks-hosted MLflow costs tens of
    seconds, and every screen in the app opens by asking for it. Started
    here it is finished long before a browser connects; left to the first
    request it lands on a user, on a threadpool this app has exactly one
    of. Best-effort by construction: it returns immediately, and an
    unreachable tracking server is logged by the sweep itself and retried
    by the next caller.
    """
    try:
        get_pipeline_executor().prewarm()
    except Exception:  # noqa: BLE001 - warming is never worth failing startup
        logger.exception("Could not warm run history at startup")
    yield


app = FastAPI(
    lifespan=lifespan,
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
    """Return the offending field names only; log the full pydantic detail."""
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
    """Log the traceback, return a redacted message — never leak internals."""
    logger.exception("Unhandled error serving %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": safe_detail(exc)})


# Mounted at the root to match the current frontend contract.
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(profile.router)
app.include_router(metadata.router)
app.include_router(estimation.router)
app.include_router(compute.router)
app.include_router(deployment.router)
app.include_router(execution.router)
app.include_router(results.router)
app.include_router(mlflow_view.router)
