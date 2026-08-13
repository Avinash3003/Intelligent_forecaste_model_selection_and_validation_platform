from pydantic import BaseModel


class ProfileRequest(BaseModel):
    """References a previously uploaded dataset to generate a basic,
    metadata-free profile from (Section 5.1.1 — automatic data profiling,
    run immediately after upload, before any column roles are selected)."""

    file_id: str


class ColumnProfile(BaseModel):
    """One column's raw, forecasting-agnostic profile.

    `dtype` is the *raw* pandas dtype (e.g. "object", "int64",
    "datetime64[ns]") rather than an interpreted semantic type. Profiling
    deliberately makes no forecasting judgement — deciding whether an
    "object" column is a usable date or target is the Validation Engine's
    job, once the user has actually assigned roles.
    """

    name: str
    dtype: str
    sample_value: str
    null_pct: str
    distinct_values: str


class ProfileResponse(BaseModel):
    dataset_name: str
    total_rows: int
    total_columns: int
    file_size_bytes: int
    file_size: str
    columns: list[ColumnProfile]


class DateRangeRequest(BaseModel):
    """References a previously uploaded dataset and the column the user has
    tentatively assigned as its date column (Metadata Mapping, Priority B) —
    the same file `ProfileRequest` reads, one column further along.
    """

    file_id: str
    date_column: str


class DateRangeResponse(BaseModel):
    """The dataset's observed date coverage for `date_column`, computed by
    the same parsing rule `DataQualityAssessor` uses for a real run
    (`forecast_engine.s02_quality.quality_assessor.parse_date_column` /
    `date_range_from_parsed`) — never a second, independent implementation.
    `available=False` (both bounds `None`) means the column had no
    parseable date values, not a fabricated range.
    """

    available: bool
    date_range_start: str | None = None
    date_range_end: str | None = None
