"""Compute selection, options and validation contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class NodeTypeOption(BaseModel):
    """One compute size ForecastIQ offers."""

    node_type_id: str
    label: str | None = None
    description: str | None = None
    category: str | None = None
    num_cores: int | None = None
    memory_mb: int | None = None
    # Filled only by validation, which reads the live workspace catalog.
    available_core_quota: int | None = None
    available: bool = True
    unavailable_reason: str | None = None


class RuntimeOption(BaseModel):
    key: str
    name: str


class ComputeOptions(BaseModel):
    """What the compute step renders. Served without calling Databricks."""

    node_types: list[NodeTypeOption] = Field(default_factory=list)
    runtimes: list[RuntimeOption] = Field(default_factory=list)
    default_node_type_id: str | None = None
    default_runtime_key: str | None = None


class ExistingComputeResponse(BaseModel):
    """The fallback compute, or why it cannot be offered."""

    available: bool
    message: str | None = None
    compute: "ExistingCompute | None" = None


class ExistingComputeValidationResult(BaseModel):
    """Whether the configured existing compute can run this workload."""

    valid: bool
    # One short, user-facing sentence. Never a raw Databricks error.
    message: str
    state: str | None = None
    # True when the cluster is stopped but will start for the run.
    starts_on_demand: bool = False
    checked_at: str | None = None


class ExistingCompute(BaseModel):
    """The already-provisioned all-purpose compute offered as the fallback."""

    cluster_id: str
    cluster_name: str
    state: str | None = None
    node_type_id: str | None = None
    runtime: str | None = None
    num_workers: int = 0
    num_cores: int | None = None
    memory_mb: int | None = None
    autotermination_minutes: int | None = None
    single_node: bool = True


class JobComputeConfig(BaseModel):
    """A per-run job compute the user asked us to create."""

    node_type_id: str
    runtime_key: str
    autoscale: bool = False
    num_workers: int = Field(0, ge=0, le=100)
    min_workers: int = Field(1, ge=0, le=100)
    max_workers: int = Field(2, ge=1, le=100)

    @model_validator(mode="after")
    def _check_worker_bounds(self) -> "JobComputeConfig":
        if self.autoscale and self.min_workers > self.max_workers:
            raise ValueError("Minimum workers cannot be greater than maximum workers.")
        return self

    # Total cores this configuration will ask the subscription for.
    def requested_cores(self, cores_per_node: int) -> int:
        workers = self.max_workers if self.autoscale else self.num_workers
        return cores_per_node * (workers + 1)


class ComputeSelection(BaseModel):
    """Which compute a run should execute on."""

    mode: Literal["new_job_compute", "existing_compute"] = "existing_compute"
    job_compute: JobComputeConfig | None = None
    cluster_id: str | None = None

    @model_validator(mode="after")
    def _check_mode(self) -> "ComputeSelection":
        if self.mode == "new_job_compute" and self.job_compute is None:
            raise ValueError("A job compute configuration is required.")
        return self


class ComputeValidationRequest(BaseModel):
    job_compute: JobComputeConfig
    # Skips the create probe; the fast metadata checks always run.
    quick: bool = False


class ComputeValidationResult(BaseModel):
    valid: bool
    # One short, user-facing sentence. Never a raw Databricks error.
    message: str
    stage: Literal["metadata", "create_probe"] = "metadata"
    probe_cluster_deleted: bool | None = None
    checked_at: str | None = None


ExistingComputeResponse.model_rebuild()
