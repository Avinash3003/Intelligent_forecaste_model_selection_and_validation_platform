from pydantic import BaseModel, ConfigDict, Field


class LLMOpsSummary(BaseModel):
    """Run-level aggregate — the same numbers `trace_summary` already
    computes (Section 13.4), renamed to the vocabulary an LLMOps view uses.
    """

    model_config = ConfigDict(protected_namespaces=())

    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    average_latency_ms: float | None = None
    estimated_cost_usd: float | None = None
    # Whether `estimated_cost_usd` is a real figure or absent because
    # pricing was never configured — distinct from a genuine $0, which the
    # engine never reports (see `LLMConfig.pricing_for`).
    cost_available: bool = False
    groundedness_rate: float | None = None
    retry_count: int = 0
    provider: str | None = None
    deployment: str | None = None
    prompt_version: str | None = None
    # Run-level LLM status: generated | partial | failed | disabled |
    # "unavailable: <reason>" — verbatim from `BusinessInsightReport.status`.
    status: str = "not_generated"


class LLMOpsAttempt(BaseModel):
    """One request attempt for a group — a group with retries has more
    than one of these; a group with none has exactly one.
    """

    model_config = ConfigDict(protected_namespaces=())

    attempt_number: int = 1
    provider: str = "none"
    deployment: str | None = None
    routing_tier: str | None = None
    request_start_time: str | None = None
    request_end_time: str | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    validation_status: str = "not_attempted"
    validation_errors: list[str] = Field(default_factory=list)
    grounding_status: str = "not_attempted"
    grounding_issues: list[str] = Field(default_factory=list)
    final_status: str = "pending"
    error: str | None = None


class LLMOpsCall(BaseModel):
    """One forecast group's LLM activity: its final outcome plus every attempt.

    Named for its place in the UI (one row per key), though a group with
    retries covers more than one HTTP request.
    """

    model_config = ConfigDict(protected_namespaces=())

    group_id: str
    # The forecasting model this insight explains (e.g. "prophet") — not
    # the LLM deployment, which is `deployment` below.
    forecast_model: str | None = None
    provider: str = "none"  # azure_openai | azure_openai_fallback | template | none
    deployment: str | None = None
    prompt_version: str | None = None
    timestamp: str | None = None

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    estimated_cost_usd: float | None = None

    retry_count: int = 0
    validation_status: str = "not_attempted"
    grounding_status: str = "not_attempted"
    final_status: str = "pending"
    error: str | None = None

    # The actual generated content (Section 13.1's structured contract) —
    # what the LLM produced, not the prompt that was sent. No raw
    # prompt/response text is stored anywhere upstream by design (see
    # `LLMCallTrace`), so there is nothing further to redact here.
    concise_summary: str | None = None
    rejection_reasons: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    confidence: float | None = None

    validation_errors: list[str] = Field(default_factory=list)
    grounding_issues: list[str] = Field(default_factory=list)

    attempts: list[LLMOpsAttempt] = Field(default_factory=list)


class LLMOpsResponse(BaseModel):
    """LLMOps observability payload for one run."""

    run_id: str
    available: bool = False
    summary: LLMOpsSummary = Field(default_factory=LLMOpsSummary)
    calls: list[LLMOpsCall] = Field(default_factory=list)


class PromptVersionUsage(BaseModel):
    """Usage aggregated across every completed run that used one prompt version.

    Every number is a sum or rate over real calls; nothing is backfilled.
    """

    model_config = ConfigDict(protected_namespaces=())

    prompt_version: str
    runs_included: int = 0

    # Usage: what was actually spent.
    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    cost_available: bool = False

    # Performance: how fast it answered.
    average_latency_ms: float | None = None

    # Quality: how good the answers were, and how often the system had to
    # compensate — computed from each group's *final* outcome (after any
    # retries), not from every intermediate attempt.
    groundedness_rate: float | None = None
    validation_pass_rate: float | None = None
    retry_rate: float | None = None
    fallback_rate: float | None = None


class PromptUsageResponse(BaseModel):
    """Every prompt version seen across completed runs, most active first."""

    versions: list[PromptVersionUsage] = Field(default_factory=list)
