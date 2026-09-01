from enum import Enum

from pydantic import BaseModel, Field


class AggregationMethod(str, Enum):
    """How a sub-monthly target is rolled up to monthly.

    The right method depends on what the target measures — SUM for flows
    (units sold), MEAN or LAST for levels (price, inventory) — which only
    the user knows, hence the explicit choice.
    """

    SUM = "sum"
    MEAN = "mean"
    LAST = "last"


class MetadataMapping(BaseModel):
    """The user's column-role selections, with no file reference attached.

    aggregation_method is chosen after validation and applies only when the
    detected grain is finer than monthly.
    """

    date_column: str
    target_column: str
    key_columns: list[str] = Field(default_factory=list)
    feature_columns: list[str] = Field(default_factory=list)
    aggregation_method: AggregationMethod = AggregationMethod.SUM


class MetadataRequest(MetadataMapping):
    """MetadataMapping plus a pointer (`file_id`) to the dataset previously
    staged via POST /upload — /metadata/validate needs the real file to
    validate against, not just the column names."""

    file_id: str


class DatasetShape(BaseModel):
    """Row/column counts of the uploaded dataset, captured once at
    interpretation time so downstream consumers don't need to re-read the
    file just to know its size."""

    rows: int
    columns: int


class ValidationStatus(str, Enum):
    """Outcome of one validation check.

    CONVERTIBLE is the important middle state: the raw type is wrong for the
    role but the values cast safely during preprocessing (text "2025-09-09"
    as a date). Rejecting those would fail datasets the platform handles
    fine, since CSVs store almost everything as text.
    """

    VALID = "Valid"
    CONVERTIBLE = "Convertible"
    INVALID = "Invalid"


class SuitabilityStatus(str, Enum):
    """Overall verdict on whether a dataset can be forecast."""

    READY = "Ready"
    WARNINGS = "Warnings"
    NOT_SUITABLE = "Not Suitable"


class NormalizedMetadataConfig(BaseModel):
    """The normalized object MetadataInterpreter produces, and the contract
    everything downstream reads instead of re-parsing frontend metadata."""

    date_column: str
    target_column: str
    key_columns: list[str]
    feature_columns: list[str]
    mode: str
    composite_key: str | None = None
    forecast_frequency: str
    unique_keys: int
    dataset_shape: DatasetShape


class ValidationCheckItem(BaseModel):
    """One row of the validation report.

    status_label lets a check show something more specific than its status —
    Frequency shows "Daily" while still reporting VALID for decision logic.
    """

    id: str
    title: str
    status: ValidationStatus
    status_label: str | None = None
    description: str


class ForecastSuitability(BaseModel):
    """The final verdict, listing every finding at once so a user is not
    fixing problems one run at a time."""

    status: SuitabilityStatus
    summary: str
    reasons: list[str] = Field(default_factory=list)


class ForecastConfigurationSummary(BaseModel):
    """Interpreted metadata for a validated configuration.

    Field names mirror the frontend's cards so no reshaping is needed.
    Frequency is exactly as detected; nothing is aggregated at this stage.
    """

    dataset_name: str
    rows: int
    columns: int
    forecast_mode: str
    forecast_frequency: str
    unique_business_keys: int

    # Whether the target must be rolled up to monthly before forecasting.
    # The backend owns this rule so the frontend never re-derives which
    # grains need aggregation: it shows the control iff this is True.
    aggregation_required: bool = False


class MetadataValidationResponse(BaseModel):
    mode: str
    normalized_config: NormalizedMetadataConfig
    checks: list[ValidationCheckItem]
    forecast_suitability: ForecastSuitability
    ready_for_deployment: bool
    configuration_summary: ForecastConfigurationSummary


class ModelAvailability(BaseModel):
    """Whether one candidate model can run on the configured execution mode.

    The frontend's model picker reads this instead of assuming every
    registered model is runnable — otherwise it offers a choice the
    environment will report Unavailable (see app/config/model_availability.py).
    """

    id: str
    available: bool
    # Populated only when available is False, so the picker can explain the
    # disabled state rather than silently hiding the option.
    reason: str | None = None


class ForecastHorizonRange(BaseModel):
    """The business rule this platform enforces on forecast horizon,
    read from app.config.run_limits -- the same bounds the deploy and
    estimate requests validate against, so a value the picker allows is
    never one the backend then rejects."""

    min_months: int
    max_months: int
    default_months: int


class ModelAvailabilityResponse(BaseModel):
    execution_mode: str
    models: list[ModelAvailability]
    # Fetched once, at the same time as model availability, because both
    # answer the same question for the Configure step: "what can this run
    # actually be." horizon stays optional so an older cached frontend
    # build (which does not read this field) keeps working unchanged.
    horizon: ForecastHorizonRange
    # ModelConfig.DEFAULT_FALLBACK_MODEL, the same value a submitted run
    # falls back to when fallback_model is omitted (see
    # build_execution_request) -- read here only so the picker can
    # pre-select the same model it will actually run with, not a second,
    # disconnected guess of what that model is.
    default_fallback_model: str
