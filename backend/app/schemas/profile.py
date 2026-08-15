from pydantic import BaseModel


class ProfileRequest(BaseModel):
    """References a previously uploaded dataset to generate a basic,
    metadata-free profile from (Section 5.1.1 — automatic data profiling,
    run immediately after upload, before any column roles are selected)."""

    file_id: str


class ColumnProfile(BaseModel):
    """One column's raw profile, with no forecasting judgement.

    dtype is the raw pandas dtype, not an interpreted type — deciding whether
    an "object" column works as a date is the Validation Engine's job.
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
    """The column's observed date coverage, using the same parsing rule a real
    run does. available=False means nothing parsed, not a fabricated range."""

    available: bool
    date_range_start: str | None = None
    date_range_end: str | None = None
