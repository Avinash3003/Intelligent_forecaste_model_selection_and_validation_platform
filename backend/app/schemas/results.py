from pydantic import BaseModel, ConfigDict, Field


class GroupOption(BaseModel):
    """One selectable business key on the Results filter bar."""

    group_id: str
    label: str


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
    period: str
    actual: float | None = None
    forecast: float | None = None
    lower: float | None = None
    upper: float | None = None
    highlight: bool = False


class DashboardInsight(BaseModel):
    """The concise, structured form of the LLM narrative that the dashboard
    renders.

    The engine emits one prose field per group (`concise_summary`), so there
    is no separate long-form narrative to defer to — the fields below are
    hard-capped server-side (see `_dashboard_insight`) purely so a future
    schema change can't push an unbounded string onto the card. The
    `paragraphs` a caller sees alongside this (via `ExplainabilityNarrative`)
    never repeats `summary`; it holds only what the card doesn't already
    show — full rejection reasons and any caveats beyond the first.
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
    """Why this model ranked where it did (Section 6.6) — the composite
    score plus every component that fed it. This is a *relative* score
    (min-max normalized against this group's other surviving candidates,
    which is exactly why it is shown here as "ranking", not reused as
    "confidence" — see `ConfidenceExplanation` for the absolute measure).
    """

    composite_score: float | None = None
    backtest_score: float | None = None
    stability_score: float | None = None
    shap_score: float | None = None
    shap_method: str | None = None
    original_backtest_rank: int | None = None
    final_rank: int | None = None


class DriftDetail(BaseModel):
    """Drift validation outcome (Sections 6.7-6.9) for this specific model.

    `evaluated=False` is a real, honest state: Final Selection stops at the
    first candidate that passes, so a lower-ranked survivor may never have
    been drift-tested at all. That is different from "evaluated and
    failed", and the two must not be collapsed into one status string.
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


class LLMTraceSummary(BaseModel):
    """Run-level LLM usage (Section 13.4) — every field the phase's
    expected report names: calls, tokens, latency, cost, prompt version,
    groundedness, retries. Sourced verbatim from the engine's
    `LLMTraceStore.summary()`, never recomputed here.
    """

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
