from pydantic import BaseModel, ConfigDict, Field


class GroupOption(BaseModel):
    """One selectable business key on the Results filter bar.

    `key_values` carries the key columns separately so the filter bar can
    offer one dropdown per column instead of one long combined list; it is
    empty for a single-series run.
    """

    group_id: str
    label: str
    key_values: dict[str, str] = Field(default_factory=dict)


class ConfidenceExplanation(BaseModel):
    """The evidence behind `ModelDecision.confidence` — never just a bare
    number. See `app/services/confidence.py` for the full derivation.
    """

    backtest_accuracy: float | None = None
    # How closely the forward forecast's variation matches the group's own
    # history. None when either side was too short to have a meaningful
    # spread — the component is then renormalized away, not scored as zero.
    forecast_stability: float | None = None
    drift_margin: float | None = None
    formula: str
    explanation: str


class ModelDecision(BaseModel):
    selected_model: str
    # None when even backtest data is unavailable — the frontend renders
    # "N/A", never a fabricated percentage.
    confidence: float | None
    confidence_explanation: ConfidenceExplanation
    ranking_position: int
    validation_status: str
    fallback_used: bool = False
    # Populated only on the fallback path — what triggered it and which
    # candidates were considered first.
    fallback_reason: str | None = None
    original_candidates: list[str] = Field(default_factory=list)


class ForecastPoint(BaseModel):
    """One point on the Actual vs Forecast chart.

    `period` is the point's identity, not its label: actual points carry
    their ISO date and forecast points carry "T1".."Tn", which is what the
    horizon selector matches on. `label` is what the axis should read —
    for forecast points that is the projected calendar date, so a long
    series does not switch from dates to opaque T-keys halfway across.
    """

    period: str
    label: str | None = None
    actual: float | None = None
    forecast: float | None = None
    lower: float | None = None
    upper: float | None = None
    highlight: bool = False
    # True on the single point where observed history ends and the forecast
    # begins. Marked here rather than inferred by the chart so the boundary
    # is the real final historical timestamp, not a position guess.
    boundary: bool = False


class DashboardInsight(BaseModel):
    """The structured LLM narrative the dashboard card renders.

    Fields are capped server-side so a schema change cannot push an unbounded
    string onto the card. The paragraphs alongside this never repeat summary —
    they hold only what the card does not already show.
    """

    summary: str | None = None  # <= 60 words
    key_reason: str | None = None  # <= 15 words
    caveat: str | None = None  # <= 25 words
    truncated: bool = False


class ExplainabilityNarrative(BaseModel):
    key_model_headline: str
    paragraphs: list[str]
    # Empty when the run had no Azure OpenAI access; the dashboard then
    # shows the status instead of a narrative.
    available: bool = False
    status: str | None = None
    insight: DashboardInsight = Field(default_factory=DashboardInsight)


class ShapDriver(BaseModel):
    feature: str
    importance: float


class UnderlyingMetrics(BaseModel):
    drift_test: str
    threshold_method: str
    threshold_value: str
    drift_score: str
    wmape: str
    rmse: str
    mae: str
    validation_result: str


class BacktestMetrics(BaseModel):
    """This model's own out-of-sample rolling-backtest metrics (Section
    6.4) — never compared against other candidates, just what it measured.
    """

    wmape: float | None = None
    rmse: float | None = None
    mae: float | None = None
    mape: float | None = None
    smape: float | None = None
    window_count: int = 0


class ForwardValidationRule(BaseModel):
    """One elimination rule's verdict (Section 6.5.3) — the exact measured
    value, not just pass/fail, so a rejection is traceable to a number.
    """

    rule_name: str
    passed: bool
    detail: str | None = None


class RankingBreakdown(BaseModel):
    """Why this model ranked where it did: the composite score and its parts.

    Relative, normalized against this group's other survivors — which is why
    it is not reused as confidence. See ConfidenceExplanation for that.
    """

    composite_score: float | None = None
    backtest_score: float | None = None
    stability_score: float | None = None
    shap_score: float | None = None
    shap_method: str | None = None
    original_backtest_rank: int | None = None
    final_rank: int | None = None


class DriftDetail(BaseModel):
    """This model's drift outcome.

    evaluated=False is a real state: selection stops at the first candidate
    that passes, so a lower-ranked survivor may never have been tested. That
    is not the same as tested and failed.
    """

    evaluated: bool = False
    algorithm: str | None = None
    statistic: float | None = None
    threshold_value: float | None = None
    threshold_method: str | None = None
    passed: bool | None = None
    detail: str | None = None


class EvaluatedModelDetail(BaseModel):
    """The complete, traceable record for one model evaluated on the
    selected group — training through final outcome. One row per model
    that was trained for this group; nothing here is a placeholder.
    """

    model: str

    training_status: str
    training_error: str | None = None

    backtest: BacktestMetrics | None = None

    forward_validation_status: str
    forward_validation_reasons: list[str] = Field(default_factory=list)
    forward_validation_rules: list[ForwardValidationRule] = Field(default_factory=list)

    ranking: RankingBreakdown | None = None
    drift: DriftDetail | None = None

    # The one-line verdict: "Selected", "Fallback Used", "Rejected — <why>",
    # "Eliminated — <why>", "Failed to train — <why>", or "Not reached".
    selection_outcome: str


class MLflowRunInfo(BaseModel):
    run_id: str | None = None
    experiment: str | None = None
    status: str | None = None
    tracking_uri: str | None = None
    models_registered: int | None = None
    # Deep link to this run in the Databricks MLflow UI, or None when one
    # cannot be built (no workspace host configured, missing experiment/run
    # id, or a run tracked to a local store). The UI shows the "Open in
    # Databricks" action only when this is set, so the rest of the MLflow
    # record still renders when the link is unavailable.
    databricks_run_url: str | None = None


class LLMTraceSummary(BaseModel):
    """Run-level LLM usage — calls, tokens, latency, cost, groundedness,
    retries — taken verbatim from the engine's trace summary."""

    call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    average_latency_ms: float | None = None
    # None when no per-call cost could be computed (pricing not
    # configured) — rendered as "unavailable", never a fabricated $0.
    estimated_cost_usd: float | None = None
    retry_count: int = 0
    groundedness_rate: float | None = None
    prompt_version: str | None = None
    token_budget: int | None = None
    token_budget_exhausted: bool = False


class ResultsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    group_id: str | None = None
    groups: list[GroupOption] = Field(default_factory=list)
    horizon_points: list[str] = Field(default_factory=list)
    # The dataset's own observed date coverage (Assess Quality,
    # computed from its date column) — not the upload or run time. Either
    # both set or both None; never a fabricated partial range.
    dataset_date_range_start: str | None = None
    dataset_date_range_end: str | None = None
    # Derived feature columns actually used to train this run's tree-based
    # models (Priority C) — `None` means every supported feature (this run
    # never mentioned the field, or explicitly accepted every default).
    derived_features: list[str] | None = None

    model_decision: ModelDecision
    # Every model trained for this group, with its full training →
    # selection trail. This is the single source the transparency panel
    # renders from — there is deliberately no second, narrower summary list.
    evaluated_models: list[EvaluatedModelDetail]
    actual_vs_forecast: list[ForecastPoint]
    explainability: ExplainabilityNarrative
    shap_drivers: list[ShapDriver] = Field(default_factory=list)
    underlying_metrics: UnderlyingMetrics
    mlflow_run: MLflowRunInfo = Field(default_factory=MLflowRunInfo)
    llm_trace: LLMTraceSummary = Field(default_factory=LLMTraceSummary)
