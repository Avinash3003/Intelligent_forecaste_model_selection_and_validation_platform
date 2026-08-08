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
    """Structured execution summary for developer debugging.

    Every field here is read from the same `PipelineExecutionResult` the
    Results dashboard renders from — nothing is recomputed, so this view
    can never disagree with the dashboard about what actually happened.
    """

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

    # The complete, unmodified result envelope this summary was built from
    # — Section 5.7's "show underlying metrics" principle applied to the
    # whole run, not just one panel, for full traceability on demand.
    raw_result: dict = Field(default_factory=dict)
