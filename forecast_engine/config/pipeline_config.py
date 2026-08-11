"""Pipeline configuration — the plug-and-play control surface for the
Forecast Engine (Section 9, "Plug-and-Play / Configuration Architecture").

Every behavioural choice the preprocessing pipeline makes is expressed here
as a value rather than baked into the code, so a deployment can change
imputation strategy, minimum history, or parallelism without touching the
engine. Later phases extend these blocks (backtesting windows, drift
thresholds, ranking weights) rather than introducing parallel config
mechanisms.

Defaults are deliberately conservative: the engine preserves raw data
characteristics unless a deployment explicitly opts into altering them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MissingValueStrategy(str, Enum):
    """How a column's missing values should be treated during preparation.

    NONE is the default because Section 6.2 requires imputation to be
    "disabled by default so raw data characteristics are preserved for
    validation stages" — later forward-validation stages need to see the
    real gaps in a series to judge a model honestly.
    """

    NONE = "none"
    DROP = "drop"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"


class DuplicateScope(str, Enum):
    """Which columns define a "duplicate" row.

    FULL_ROW only removes rows that are identical everywhere. KEY_AND_DATE
    is stricter: it treats two rows sharing the same business key and
    timestamp as duplicates even if their target values differ, which is
    the correct reading for a time series (one observation per key, per
    period) but discards data, so it is not the default.
    """

    FULL_ROW = "full_row"
    KEY_AND_DATE = "key_and_date"


# Recognised sampling grains. The engine detects and records these but
# never converts between them in this phase — grain normalization is a
# separate, later ingestion concern (Section 6.2).
SUPPORTED_FREQUENCIES: tuple[str, ...] = ("Daily", "Weekly", "Monthly", "Quarterly", "Yearly")

# Label used when the spacing between timestamps is too inconsistent to
# map onto any supported grain.
IRREGULAR_FREQUENCY = "Irregular"

# Median gap between consecutive observed dates (in days) -> grain label.
# Bands are generous so a handful of missing periods or weekend-skipping
# data does not misclassify an otherwise regular series.
FREQUENCY_BANDS: tuple[tuple[float, float, str], ...] = (
    (0, 1, "Daily"),
    (2, 9, "Weekly"),
    (10, 45, "Monthly"),
    (46, 135, "Quarterly"),
    (136, 400, "Yearly"),
)


# Detected grain -> pandas frequency alias, used to reconstruct the expected
# timeline when counting missing timestamps. Grains that cannot be mapped
# (Irregular/Unknown) simply skip that check.
PANDAS_FREQUENCY_ALIASES: dict[str, str] = {
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "MS",
    "Quarterly": "QS",
    "Yearly": "YS",
}


@dataclass(frozen=True)
class PreprocessingConfig:
    """Controls how the raw dataset is cleaned before series are built."""

    # Target and feature columns get separate strategies. A missing target is
    # a missing *observation* — the row carries nothing to forecast — so the
    # default drops it. Features are deliberately left untouched: filling
    # regressors is feature-engineering work, which this phase excludes.
    target_missing_value_strategy: MissingValueStrategy = MissingValueStrategy.DROP
    feature_missing_value_strategy: MissingValueStrategy = MissingValueStrategy.NONE

    drop_duplicate_rows: bool = True
    duplicate_scope: DuplicateScope = DuplicateScope.FULL_ROW

    # Rows whose date cannot be parsed cannot be placed on a timeline, so
    # they are dropped rather than silently carried as NaT.
    drop_unparseable_dates: bool = True

    # Values that cannot be coerced to a number become NaN and are then
    # handled by target_missing_value_strategy.
    coerce_target_to_numeric: bool = True


@dataclass(frozen=True)
class ConversionConfig:
    """Controls the type standardization applied to the curated dataset.

    Conversion is attempted rather than assumed: a column that survives at
    least `minimum_success_ratio` of its values is converted, and one that
    does not is rejected outright. Rejecting is deliberate — a target that is
    really categorical ("Apple", "High") must never reach forecasting.
    """

    convert_date_column: bool = True
    convert_target_column: bool = True

    # Fraction of non-null values that must convert for the column to be
    # accepted. Below 1.0 so a few stray entries don't reject an otherwise
    # clean column; those stray rows are then dropped as invalid.
    minimum_success_ratio: float = 0.95

    # Interpret ambiguous DD/MM vs MM/DD dates as day-first when set.
    dayfirst: bool = False


@dataclass(frozen=True)
class AggregationConfig:
    """Controls the roll-up of sub-monthly data to the monthly grain.

    The *target's* method is not configured here — it is the user's choice,
    carried on the ForecastConfiguration. These settings govern how feature
    columns come along for the ride, which the design describes as a
    column-level rule rather than one global default (Section 6.2).
    """

    enabled: bool = True

    # Applied to numeric feature columns with no explicit override. Mean
    # rather than sum, because most regressors (price, temperature) are
    # levels whose monthly total would be meaningless.
    default_feature_method: str = "mean"

    # Per-column overrides, e.g. {"promo_days": "sum", "inventory": "last"}.
    feature_methods: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds that decide whether a dataset is fit to forecast."""

    # Above this share of missing target values the dataset is unusable —
    # dropping the affected rows would discard too much history to trust.
    max_target_missing_ratio: float = 0.10

    # Absolute floor below which forecasting is not meaningful at all.
    min_observations: int = 12

    # Missing-timestamp detection rebuilds an expected timeline per group,
    # which is linear in group count. Skipped above this many groups so
    # assessment stays fast on high-cardinality datasets.
    detect_missing_timestamps: bool = True
    max_groups_for_timestamp_scan: int = 5000


@dataclass(frozen=True)
class CuratedStorageConfig:
    """Where the curated dataset is written.

    Only the location and format live here. The storage *mechanism* is a
    swappable backend (see forecast_engine/storage), so moving from local
    disk to ADLS/Blob later changes the backend, never this config's meaning
    or any business logic that depends on it.
    """

    enabled: bool = True
    root_dir: str = "curated"
    file_format: str = "csv"


@dataclass(frozen=True)
class ModelStorageConfig:
    """Where each forecast key's winning fitted model is written.

    Same shape and the same reasoning as `CuratedStorageConfig`: only the
    location lives here, and a caller running in the cloud passes an
    already-resolved absolute path so no storage decision reaches the
    engine. Only the model that won Final Production Model Selection is
    persisted — candidates that lost are not.
    """

    enabled: bool = True
    root_dir: str = "models"


@dataclass(frozen=True)
class ForecastExportConfig:
    """Where the run's exported forecast values are written.

    A business-facing export — the actual forecast values every group
    produced (dates, point forecast, bounds) — separate from the curated
    dataset (the *input* the models trained on) and from the winning
    `.pkl` (the *model*, not its output). Same shape as the other storage
    configs: only the location lives here.
    """

    enabled: bool = True
    root_dir: str = "forecasts"


@dataclass(frozen=True)
class ArtifactsMirrorConfig:
    """Where a copy of this run's business insights and LLM trace is
    written outside MLflow.

    MLflow remains the primary, authoritative record (Section 6.13) —
    this is a duplicate for direct blob access, not a replacement. Only
    already-serialized JSON already produced for the run summary is
    copied here; nothing is recomputed or re-rendered.
    """

    enabled: bool = True
    root_dir: str = "artifacts"


@dataclass(frozen=True)
class GroupingConfig:
    """Controls how business keys become forecasting groups."""

    # Section 10: the platform's ≥70% accuracy target assumes at least 24
    # periods of history. Groups below this are flagged, not dropped, so
    # the caller decides what to do with thin keys.
    min_observations_per_group: int = 24
    drop_groups_below_minimum: bool = False

    # Joins multi-column key values into one readable group id, e.g.
    # store_id=1 + item=3 -> "1 | 3".
    key_separator: str = " | "

    # Group id used when no key columns are selected (single-series mode).
    single_series_group_id: str = "__single_series__"


@dataclass(frozen=True)
class ExecutionConfig:
    """Controls how work is distributed across forecasting groups.

    Parallelism is off by default: this phase performs no model training,
    so the per-group work is cheap and the coordination overhead would
    dominate. The knobs exist because the Model Training Engine in the
    next phase will fan out across thousands of keys (Section 8.4).
    """

    parallel_enabled: bool = False
    max_workers: int = 4


@dataclass(frozen=True)
class ForecastDefaults:
    """Run-level forecasting defaults, consumed by later phases.

    Declared here so the Model Training Engine reads them from the same
    config object the preprocessing stages already use.
    """

    horizon: int = 12
    confidence_level: float = 0.95


@dataclass(frozen=True)
class PipelineConfig:
    """Root configuration object carried through the whole pipeline."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    conversion: ConversionConfig = field(default_factory=ConversionConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    curated_storage: CuratedStorageConfig = field(default_factory=CuratedStorageConfig)
    model_storage: ModelStorageConfig = field(default_factory=ModelStorageConfig)
    forecast_export: ForecastExportConfig = field(default_factory=ForecastExportConfig)
    artifacts_mirror: ArtifactsMirrorConfig = field(default_factory=ArtifactsMirrorConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    forecast_defaults: ForecastDefaults = field(default_factory=ForecastDefaults)

    # Return the standard configuration used when a caller supplies none
    @classmethod
    def default(cls) -> "PipelineConfig":
        return cls()

    # Build a PipelineConfig from a nested dict, unknown keys ignored
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineConfig":
        return cls(
            preprocessing=_build_block(PreprocessingConfig, payload.get("preprocessing")),
            conversion=_build_block(ConversionConfig, payload.get("conversion")),
            aggregation=_build_block(AggregationConfig, payload.get("aggregation")),
            quality=_build_block(QualityConfig, payload.get("quality")),
            curated_storage=_build_block(CuratedStorageConfig, payload.get("curated_storage")),
            model_storage=_build_block(ModelStorageConfig, payload.get("model_storage")),
            forecast_export=_build_block(ForecastExportConfig, payload.get("forecast_export")),
            artifacts_mirror=_build_block(ArtifactsMirrorConfig, payload.get("artifacts_mirror")),
            grouping=_build_block(GroupingConfig, payload.get("grouping")),
            execution=_build_block(ExecutionConfig, payload.get("execution")),
            forecast_defaults=_build_block(ForecastDefaults, payload.get("forecast_defaults")),
        )


# Instantiate one config block, keeping only keys it declares
def _build_block(block_type: type, values: dict[str, Any] | None) -> Any:
    # Filtering unknown keys means an operator can leave forward-looking or
    # commented-out settings in a config file without breaking the run.
    if not values:
        return block_type()

    allowed = {f.name for f in block_type.__dataclass_fields__.values()}
    return block_type(**{key: value for key, value in values.items() if key in allowed})
