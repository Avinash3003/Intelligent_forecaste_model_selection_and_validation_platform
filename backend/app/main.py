from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import deployment, execution, health, metadata, mlflow_view, profile, results, upload
from app.config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Backend foundation for the Intelligent Forecast Model Selection & Validation Platform.",
    version="0.1.0",
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

# Routed at the root today (matching the current frontend contract) rather
# than under settings.api_v1_prefix; switch to the prefix once the frontend
# is updated to call versioned paths.
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(profile.router)
app.include_router(metadata.router)
app.include_router(deployment.router)
app.include_router(execution.router)
app.include_router(results.router)
app.include_router(mlflow_view.router)
