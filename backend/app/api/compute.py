from fastapi import APIRouter, Depends

from app.auth.dependencies import require
from app.auth.models import Permission, Principal
from app.schemas.compute import (
    ComputeOptions,
    ComputeValidationRequest,
    ComputeValidationResult,
    ExistingComputeListResponse,
    ExistingComputeValidationRequest,
    ExistingComputeValidationResult,
)
from app.services.compute_service import ComputeService

router = APIRouter(tags=["Compute"])
compute_service = ComputeService()


@router.get("/compute/options", response_model=ComputeOptions, summary="Compute sizes ForecastIQ offers")
def compute_options(
    principal: Principal = Depends(require(Permission.FORECAST_RUN)),
) -> ComputeOptions:
    # Project presets, so the wizard step renders without waiting on Databricks.
    return compute_service.get_options()


@router.get(
    "/compute/existing",
    response_model=ExistingComputeListResponse,
    summary="Every all-purpose cluster in the workspace the picker can offer",
)
def existing_compute(
    principal: Principal = Depends(require(Permission.FORECAST_RUN)),
) -> ExistingComputeListResponse:
    return compute_service.list_existing_compute()


@router.post(
    "/compute/existing/validate",
    response_model=ExistingComputeValidationResult,
    summary="Check the selected existing compute can run this workload",
)
def validate_existing_compute(
    request: ExistingComputeValidationRequest,
    principal: Principal = Depends(require(Permission.FORECAST_RUN)),
) -> ExistingComputeValidationResult:
    # Reads only; never starts, resizes or modifies the cluster.
    return compute_service.validate_existing_compute(request.cluster_id)


@router.post(
    "/compute/validate",
    response_model=ComputeValidationResult,
    summary="Validate a requested job compute against Databricks",
)
def validate_compute(
    request: ComputeValidationRequest,
    principal: Principal = Depends(require(Permission.FORECAST_RUN)),
) -> ComputeValidationResult:
    # An invalid configuration is a normal answer, not an HTTP error.
    return compute_service.validate(request.job_compute, quick=request.quick)
