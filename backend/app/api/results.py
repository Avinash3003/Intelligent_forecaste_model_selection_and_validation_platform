"""Forecast Insights Dashboard (Sections 5.4–5.7).

Serves the real result of a completed run. A run that does not exist, or
has not finished, returns an explicit error rather than a placeholder
dashboard — there is no dummy payload behind this route.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import require
from app.auth.models import Permission, Principal
from app.orchestration.exceptions import RunNotReadyError, UnknownRunError
from app.schemas.debug import DebugSummary
from app.schemas.dataset_preview import DatasetPreview
from app.schemas.llmops import LLMOpsResponse
from app.services.dataset_preview_service import (
    DatasetPreviewService,
    get_dataset_preview_service,
)
from app.schemas.results import ResultsResponse
from app.services.debug_service import DebugService
from app.services.llmops_service import LLMOpsService, get_llmops_service
from app.services.result_service import ResultService

router = APIRouter(prefix="/results", tags=["Results"])
result_service = ResultService()
debug_service = DebugService()


@router.get("/{run_id}", response_model=ResultsResponse, summary="Get the Forecast Insights Dashboard for a run")
def get_results(
    run_id: str,
    group_id: str | None = Query(None, description="Business key to display; defaults to the first group."),
    principal: Principal = Depends(require(Permission.RESULTS_READ)),
) -> ResultsResponse:
    try:
        return result_service.get_results(run_id, group_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail="That run could not be found.") from exc
    except RunNotReadyError as exc:
        raise HTTPException(status_code=409, detail="This run has not finished yet.") from exc


@router.get(
    "/{run_id}/debug",
    response_model=DebugSummary,
    summary="Developer debugging mode — structured execution summary for a run",
)
def get_debug_summary(
    run_id: str,
    principal: Principal = Depends(require(Permission.MODEL_INSPECT)),
) -> DebugSummary:
    # Deliberately does not require COMPLETED — a run that is still
    # executing or that failed is exactly what a developer most wants to
    # inspect; only an unknown run_id is an error here.
    try:
        return debug_service.get_debug_summary(run_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail="That run could not be found.") from exc


@router.get(
    "/{run_id}/llmops",
    response_model=LLMOpsResponse,
    summary="LLMOps observability — per-call LLM trace for a run",
)
def get_llmops(
    run_id: str,
    service: LLMOpsService = Depends(get_llmops_service),
    principal: Principal = Depends(require(Permission.MODEL_INSPECT)),
) -> LLMOpsResponse:
    # Same rationale as /debug: a developer inspecting LLM activity on a
    # still-running or failed run is the normal case, not an error, so this
    # does not require COMPLETED. Only an unknown run_id is a 404.
    try:
        return service.get_llmops(run_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail="That run could not be found.") from exc


@router.get("/{run_id}/dataset-preview", response_model=DatasetPreview)
def get_dataset_preview(
    run_id: str,
    page: int = Query(1, ge=1, description="1-indexed page number."),
    page_size: int = Query(50, ge=1, le=200, description="Rows per page (max 200)."),
    service: DatasetPreviewService = Depends(get_dataset_preview_service),
    principal: Principal = Depends(require(Permission.DATASET_READ)),
) -> DatasetPreview:
    """One page of the curated dataset this run trained on.

    Page 1 is rows 1-`page_size`, page 2 is the next `page_size`, and so on —
    plain offset paging rather than a byte-range cursor, since the curated
    file (post-preprocessing, one row per key per period) is small enough to
    parse once and slice per page.

    Never raises for a missing file: no curated dataset for this run makes
    the preview unavailable, not the whole Results page, so this returns
    `available=False` with the reason instead of an error status.
    """
    return service.get_preview(run_id, page=page, page_size=page_size)
