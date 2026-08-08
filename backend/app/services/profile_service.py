"""Dataset Profiling Service — the "Basic Dataset Inspection" summary shown
immediately after upload (Section 5.1.1), before any metadata is mapped.

Strictly descriptive: it reports what the file physically contains — raw
pandas dtypes, sample values, null ratios, cardinality — and makes no
forecasting judgement whatsoever. Deciding whether a column can serve as a
date, target or key belongs to the Validation Engine and only becomes
answerable once the user has assigned roles.

Because it never interprets, this service is inherently dataset-agnostic:
no column name is referenced anywhere.
"""

import pandas as pd

from app.schemas.profile import ColumnProfile, ProfileResponse


def format_file_size(size_bytes: int) -> str:
    """Render a byte count as a human-readable size.

    Args:
        size_bytes: File size in bytes.

    Returns:
        A short string such as "2.1 MB", for display on the inspection card.
    """
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
        """Profile every column in `dataframe`.

        Args:
            dataframe: The uploaded dataset, already loaded by DatasetLoader.
            dataset_name: Original filename, for display.
            file_size_bytes: Size of the staged file on disk.

        Returns:
            A ProfileResponse with one ColumnProfile per column. Called by
            the /profile route immediately after a successful upload.
        """
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
        """Describe a single column without interpreting it.

        The dtype is reported exactly as pandas inferred it on read, so the
        user sees the real storage type ("object" for text-encoded dates,
        for instance) rather than a guess that might disagree with what the
        Validation Engine later concludes.
        """
        non_null = series.dropna()

        return ColumnProfile(
            name=str(series.name),
            dtype=str(series.dtype),
            sample_value=str(non_null.iloc[0]) if not non_null.empty else "—",
            null_pct=f"{(series.isna().mean() * 100):.1f}%",
            distinct_values=f"{non_null.nunique():,}",
        )
