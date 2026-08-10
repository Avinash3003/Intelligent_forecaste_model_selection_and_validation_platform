"""Schemas for the pre-run cost/time estimate.

Every number here is derived from the actual uploaded dataset and the
actual selected configuration — never a fixed "2-7 minutes" placeholder.
See `app/services/estimation_service.py` for how each field is computed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.metadata import MetadataMapping


class EstimationRequest(BaseModel):
    """The same configuration a run would be submitted with.

    Takes the whole shape rather than a few scalars so the estimate is
    computed from exactly what will execute — an estimate derived from a
    different set of inputs than the run is worse than none.
    """

    file_id: str
    metadata: MetadataMapping
    selected_models: list[str] = Field(default_factory=list)
    horizon: int = Field(12, ge=6, le=60)


class EstimateComponent(BaseModel):
    """One line of the estimate's breakdown, so the total is explainable."""

    label: str
    detail: str


class DatasetMetadataSummary(BaseModel):
    """What the estimate was computed from — read straight from the
    uploaded file, not guessed. Shown to the user so "why will this take
    that long" is answerable from this block alone.
    """

    rows: int
    columns: int
    date_column: str
    target_column: str
    feature_columns: list[str] = Field(default_factory=list)
    key_columns: list[str] = Field(default_factory=list)
    unique_keys: int
    date_grain: str
    # Longest single group's observation count — the platform's real
    # per-key history depth, which is what min-observation gates (tuning,
    # TFT, backtesting) are actually measured against.
    history_length_periods: int
    # None when the target column could not be read as numeric at
    # profiling time — never a fabricated 0%.
    missingness_pct: float | None = None
    forecast_horizon: int
    selected_models: list[str] = Field(default_factory=list)


class WorkloadEstimate(BaseModel):
    """The actual amount of work this configuration implies, derived from
    keys x models x the pipeline's own per-pair fit structure — not a
    single blended "it depends" number.
    """

    model_config = ConfigDict(protected_namespaces=())

    forecast_groups: int
    models_per_group: int
    # keys x models: the base unit of work every later stage scales from.
    model_evaluations: int
    # Backtest folds actually evaluated (rolling/expanding windows x
    # evaluations) — each one is a full model fit (Section 6.4, "honest
    # refitting"), so this is the single largest driver of Evaluation time.
    backtest_windows: int
    # One per surviving model x group (Section 6.5) — reuses Training's own
    # fit rather than a second one (see forecast_engine's
    # ForwardForecastGenerator), so this does NOT add a fit of its own.
    forward_validation_forecasts: int
    # Whether hyperparameter tuning will actually run for at least one
    # (group, model) pair, and how many pairs clear the minimum-history bar.
    tuning_eligible_pairs: int
    # SHAP/permutation-importance computations — one per surviving model.
    shap_computations: int
    # One structured LLM call per forecast group (Section 6.1 Task 10),
    # when Azure OpenAI is configured; otherwise the deterministic
    # template path runs instead, at effectively zero marginal time.
    llm_calls: int


class CostBreakdown(BaseModel):
    """Databricks compute and LLM cost, kept and shown separately (Section
    8.5) since they scale on entirely different axes — one per
    cluster-minute, the other per key/token.

    Every `*_available` flag is false, and the paired low/high fields are
    null, whenever the underlying rate is not configured — never a
    fabricated figure on either line.
    """

    databricks_cost_low: float | None = None
    databricks_cost_high: float | None = None
    databricks_cost_available: bool = False

    llm_cost_low: float | None = None
    llm_cost_high: float | None = None
    llm_cost_available: bool = False

    total_cost_low: float | None = None
    total_cost_high: float | None = None
    total_cost_available: bool = False

    currency: str = "USD"


class EstimationResponse(BaseModel):
    """A range, never a single number.

    Runtime depends on series length, model mix and how many keys clear the
    minimum-history bar, none of which is knowable before the run. A range
    with a stated basis is honest; a precise-looking single figure would
    not be.
    """

    dataset: DatasetMetadataSummary
    workload: WorkloadEstimate
    cost: CostBreakdown

    estimated_minutes_low: float
    estimated_minutes_high: float
    estimated_duration_label: str

    execution_backend: str
    breakdown: list[EstimateComponent] = Field(default_factory=list)
    basis: str

    # "heuristic" (no usable run history yet) or "historical (N runs)" —
    # named explicitly per the spec's requirement to distinguish the two,
    # rather than silently blending them.
    calibration_basis: str
