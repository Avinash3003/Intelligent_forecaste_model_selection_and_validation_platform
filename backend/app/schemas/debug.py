from pydantic import BaseModel, ConfigDict, Field


class StatusCount(BaseModel):
    status: str
    count: int


class GroupSelectionSummary(BaseModel):
    group_id: str
    winner_model: str | None = None
    selection_status: str
    fallback_used: bool = False


class ArtifactLocation(BaseModel):
    name: str
    path: str


class DebugSummary(BaseModel):
    """Execution summary for debugging, read from the same result the
    dashboard renders — so the two can never disagree."""

    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    job_status: str

    dataset_path: str | None = None
    dataset_name: str | None = None
    frequency: str | None = None
    mode: str | None = None
    aggregation_method: str | None = None

    forecast_group_count: int | None = None
    series_count: int | None = None
    selected_models: list[str] = Field(default_factory=list)
    # Best-effort: the run-level fallback configuration is only visible on
    # this payload via a group that actually used it. A run where every
    # group had a clear ranked winner carries no fallback record at all —
    # that is not missing data, the fallback simply never ran.
    fallback_model: str | None = None

    training_summary: list[StatusCount] = Field(default_factory=list)
    evaluation_summary: list[StatusCount] = Field(default_factory=list)
    winner_selection: list[GroupSelectionSummary] = Field(default_factory=list)

    mlflow_run_id: str | None = None
    mlflow_experiment: str | None = None
    mlflow_tracking_uri: str | None = None
    models_registered: int | None = None
    artifact_locations: list[ArtifactLocation] = Field(default_factory=list)

    execution_duration_seconds: float | None = None
    stages: list[dict] = Field(default_factory=list)

    # The Ray key-level execution telemetry (executor, ray_cpus,
    # max_concurrent_keys, wall_seconds, key_spans), lifted out of
    # `raw_result.execution_summary.metadata` into a named field.
    #
    # It already reached the browser inside `raw_result` — but only at a
    # four-level-deep path into a payload whose whole purpose is being an
    # unmodified dump. Naming it here lets the parallel-execution view read
    # one documented field instead of coupling itself to the envelope's
    # internal shape.
    #
    # None for a sequential run, a run that predates this telemetry, or one
    # that has not reached Train Models yet — all three mean "nothing to
    # show", never an error.
    key_execution: dict | None = None

    # The complete, unmodified result envelope this summary was built from
    # — Section 5.7's "show underlying metrics" principle applied to the
    # whole run, not just one panel, for full traceability on demand.
    raw_result: dict = Field(default_factory=dict)
