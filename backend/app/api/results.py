"""Forecast Insights Dashboard (Sections 5.4–5.7).

Serves the real result of a completed run. A run that does not exist, or
has not finished, returns an explicit error rather than a placeholder
dashboard — there is no dummy payload behind this route.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.orchestration.exceptions import RunNotReadyError, UnknownRunError
from app.schemas.debug import DebugSummary
from app.schemas.dataset_preview import DatasetPreview
from app.services.dataset_preview_service import (
    DatasetPreviewService,
    get_dataset_preview_service,
)
from app.schemas.results import ResultsResponse
from app.services.debug_service import DebugService
from app.services.result_service import ResultService

router = APIRouter(prefix="/results", tags=["Results"])
result_service = ResultService()
debug_service = DebugService()


@router.get("/{run_id}", response_model=ResultsResponse, summary="Get the Forecast Insights Dashboard for a run")
def get_results(
    run_id: str,
    group_id: str | None = Query(None, description="Business key to display; defaults to the first group."),
) -> ResultsResponse:
    try:
        return result_service.get_results(run_id, group_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RunNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/{run_id}/debug",
    response_model=DebugSummary,
    summary="Developer debugging mode — structured execution summary for a run",
)
def get_debug_summary(run_id: str) -> DebugSummary:
    # Deliberately does not require COMPLETED — a run that is still
    # executing or that failed is exactly what a developer most wants to
    # inspect; only an unknown run_id is an error here.
    try:
        return debug_service.get_debug_summary(run_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{run_id}/dataset-preview", response_model=DatasetPreview)
def get_dataset_preview(
    run_id: str,
    service: DatasetPreviewService = Depends(get_dataset_preview_service),
) -> DatasetPreview:
    """The head of the file this run was built from.

    Never raises for a missing file: an unreadable dataset makes the preview
    unavailable, not the whole Results page, so this returns `available=False`
    with the reason instead of an error status.
    """
    return service.get_preview(run_id)
