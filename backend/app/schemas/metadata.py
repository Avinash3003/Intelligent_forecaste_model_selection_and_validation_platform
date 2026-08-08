from enum import Enum

from pydantic import BaseModel, Field


class AggregationMethod(str, Enum):
    """How a sub-monthly target is rolled up to the monthly grain.

    All forecasting runs at month level, so Daily/Weekly datasets are
    aggregated during preprocessing. The right method depends on what the
    target measures — SUM for flow quantities (units sold), MEAN or LAST
    for stock-type levels (price, inventory) — which only the user knows,
    hence the explicit choice.
    """

    SUM = "sum"
    MEAN = "mean"
    LAST = "last"


class MetadataMapping(BaseModel):
    """The column-role selections a user makes in Metadata Mapping, with no
    file reference attached. Reused wherever only the selections themselves
    matter — e.g. nested inside DeploymentRequest, which already carries its
    own top-level `file_id`.

    `aggregation_method` is chosen after validation and only applies when
    the detected grain is finer than monthly; it is ignored otherwise.
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
    """Outcome of a single metadata validation.

    CONVERTIBLE is the important middle state: a column whose raw storage
    type is wrong for its assigned role, but whose values can be safely
    converted during preprocessing (text "2025-09-09" used as a date, text
    "100" used as a target). Rejecting these outright would fail datasets
    the platform can handle perfectly well, since CSV uploads store almost
    everything as text.
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
    """The single normalized object produced by MetadataInterpreter.

    This is the contract the rest of the platform is built around: the
    ValidationEngine consumes it today, and the future Databricks pipeline
    will consume it directly instead of re-parsing frontend metadata
    (Section 4 — the orchestration/results contracts decouple layers).
    """

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
    """One row of the Metadata Validation Report.

    `status_label` lets a check display something more specific than its
    status on the badge — the Frequency check shows the detected grain
    ("Daily") while still reporting status VALID for decision logic.
    """

    id: str
    title: str
    status: ValidationStatus
    status_label: str | None = None
    description: str


class ForecastSuitability(BaseModel):
    """Final verdict shown after the validation report.

    `reasons` lists every blocking or cautionary finding, so a user sees
    all problems at once rather than fixing them one run at a time.
    """

    status: SuitabilityStatus
    summary: str
    reasons: list[str] = Field(default_factory=list)


class ForecastConfigurationSummary(BaseModel):
    """Interpreted metadata accompanying a validated configuration.

    Field names mirror the frontend's summary cards so the payload can be
    consumed with no reshaping. Frequency is reported exactly as detected —
    no normalization or aggregation is applied or implied at this stage.
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
