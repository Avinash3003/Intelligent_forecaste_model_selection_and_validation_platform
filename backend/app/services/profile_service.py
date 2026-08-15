"""The dataset summary shown right after upload, before any mapping.

profile() is purely descriptive — dtypes, samples, null ratios, cardinality
— and never judges a column by name. compute_date_range() is separate
because it runs one step later, once the user has named a date column.
"""

import pandas as pd

from app.schemas.profile import ColumnProfile, ProfileResponse


def format_file_size(size_bytes: int) -> str:
    """Render a byte count as a short size string such as "2.1 MB"."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            precision = 0 if unit == "B" else 1
            return f"{size:.{precision}f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


class ProfileService:
    """Builds a ProfileResponse from a DataFrame."""

    def profile(self, dataframe: pd.DataFrame, dataset_name: str, file_size_bytes: int) -> ProfileResponse:
        """One ColumnProfile per column, for the /profile route after upload."""
        columns = [self._profile_column(dataframe[column]) for column in dataframe.columns]

        return ProfileResponse(
            dataset_name=dataset_name,
            total_rows=int(dataframe.shape[0]),
            total_columns=int(dataframe.shape[1]),
            file_size_bytes=file_size_bytes,
            file_size=format_file_size(file_size_bytes),
            columns=columns,
        )

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        """Describe one column without interpreting it.

        The dtype is exactly what pandas inferred, so the user sees the real
        storage type rather than a guess validation might later contradict.
        """
        non_null = series.dropna()

        return ColumnProfile(
            name=str(series.name),
            dtype=str(series.dtype),
            sample_value=str(non_null.iloc[0]) if not non_null.empty else "—",
            null_pct=f"{(series.isna().mean() * 100):.1f}%",
            distinct_values=f"{non_null.nunique():,}",
        )


def compute_date_range(series: pd.Series) -> tuple[str | None, str | None]:
    """A column's real date coverage, or (None, None) if nothing parses.

    Numeric and boolean columns are refused rather than parsed, since pandas
    would read integers as nanosecond epochs and report nonsense. Duplicated
    from the engine's quality assessor on purpose: the two are separately
    deployed processes, and the backend never imports engine code.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        parsed = pd.Series(pd.NaT, index=series.index)
    else:
        parsed = pd.to_datetime(series, errors="coerce")

    valid = parsed.dropna()
    if valid.empty:
        return None, None
    return valid.min().isoformat(), valid.max().isoformat()
