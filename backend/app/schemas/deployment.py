from pydantic import BaseModel, Field

from app.schemas.metadata import MetadataMapping


class DeploymentRequest(BaseModel):
    file_id: str | None = None
    dataset_name: str | None = None
    metadata: MetadataMapping
    selected_models: list[str] = Field(default_factory=list)
    # Model used when every evaluated model fails validation. Must be one of
    # `selected_models`; None falls back to the engine's configured default.
    fallback_model: str | None = None
    # Months to forecast forward (Section 3: minimum 12-month horizon,
    # extended by this platform down to 6 and up to 60).
    horizon: int = Field(12, ge=6, le=60)


class DeploymentResponse(BaseModel):
    run_id: str
    status: str
    message: str


class StageStatus(BaseModel):
    label: str
    status: str
    # ISO 8601 UTC, as recorded by the engine (forecast_engine/core/
    # pipeline_context.py's StageRecord) — the frontend converts to IST for
    # display, exactly like every other timestamp in the app. None while a
    # stage has not started yet, or has started but not finished.
    started_at: str | None = None
    completed_at: str | None = None


class DeploymentStatus(BaseModel):
    id: str
    dataset: str
    status: str
    start_time: str
    duration: str
    progress: int
    current_stage: str
    estimated_remaining: str
    stages: list[StageStatus]
    # Populated only for a failed run, so the UI can show why it failed
    # rather than a generic message.
    error: str | None = None
