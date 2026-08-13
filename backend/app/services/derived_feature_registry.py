"""The authoritative list of derived feature columns a deployment request
may select (Priority C) — mirrors `forecast_engine.config.
derived_features_config.SUPPORTED_FEATURE_IDS` exactly.

Kept as a small, separate duplicate rather than an import of that module:
the backend and forecast_engine are two independently deployable processes
with separate dependencies (the backend invokes forecast_engine only as an
external subprocess/job, never in-process — see LocalRunner/
DatabricksRunner), the same reasoning `services/profile_service.py`'s
`compute_date_range()` already documents. The engine itself validates
again, defensively, regardless of what passes here — this is the fast,
user-facing rejection, not the only one.
"""

from __future__ import annotations

SUPPORTED_LAGS: tuple[int, ...] = (1, 2, 3, 12)
SUPPORTED_ROLLING_WINDOWS: tuple[int, ...] = (3, 6)
SUPPORTED_CALENDAR_FEATURES: tuple[str, ...] = ("month", "quarter")

SUPPORTED_DERIVED_FEATURE_IDS: frozenset[str] = frozenset(
    [f"lag_{n}" for n in SUPPORTED_LAGS]
    + [f"rolling_mean_{n}" for n in SUPPORTED_ROLLING_WINDOWS]
    + list(SUPPORTED_CALENDAR_FEATURES)
)


def validate_derived_features(requested: list[str]) -> list[str]:
    """The requested ids that are NOT in the authoritative registry —
    empty means every id was valid. Never silently drops or substitutes;
    the caller decides what a non-empty result means (a 400, typically)."""
    return [feature_id for feature_id in requested if feature_id not in SUPPORTED_DERIVED_FEATURE_IDS]
