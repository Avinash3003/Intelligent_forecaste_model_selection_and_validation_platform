"""Pipeline Context — the state object carried through every engine stage.

Rather than threading a growing list of arguments between stages, each
stage reads what it needs from this context and writes its output back
onto it. That gives the pipeline three properties the platform needs:

  * Extensibility — a later phase (training, backtesting, drift) adds a
    field instead of changing every stage signature.
  * Auditability — the context accumulates a timed record of every stage,
    which becomes the MLflow run trail once that integration lands
    (Section 6.11).
  * Isolation — one context represents exactly one run, keyed by run_id.

The context is deliberately mutable; it is the one place in the engine
where accumulating state is the point.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from forecast_engine.config.pipeline_config import PipelineConfig
from forecast_engine.core.forecast_configuration import ForecastConfiguration, ForecastMode

if TYPE_CHECKING:
    # Imported for typing only — importing at runtime would create a cycle,
    # since the preprocessing modules import this package's core types.
    from forecast_engine.s01_preprocessing.group_generator import ForecastGroup
    from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
    from forecast_engine.s02_quality.quality_report import PreprocessingSummary, QualityReport
    from forecast_engine.s04_training.model_trainer import TrainingReport
    from forecast_engine.s06_evaluation.evaluation_report import EvaluationReport
    from forecast_engine.s07_explainability.explainability_report import ExplainabilityReport
    from forecast_engine.s08_ranking.ranking_report import RankingReport
    from forecast_engine.s10_selection.selection_report import ProductionSelectionReport
    from forecast_engine.s11_llm.insight_report import BusinessInsightReport
    from forecast_engine.s12_tracking.tracking_report import TrackingResult


class StageStatus(str, Enum):
    """Lifecycle state of a single pipeline stage."""

    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"


@dataclass
class StageRecord:
    """Timed record of one pipeline stage's execution.

    Collected for every stage so a completed run can explain where its
    time went and which stage failed — the same per-stage isolation the
    DAG provides in production (Section 6.1).
    """

    name: str
    status: str = StageStatus.RUNNING.value
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    detail: str | None = None

    # Wall-clock duration, or None while the stage is still running
    @property
    def duration_seconds(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    # Serialize for logging and run artifacts
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "completed_at": self.completed_at.isoformat(timespec="seconds") if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "detail": self.detail,
        }


@dataclass
class PipelineContext:
    """Carries one run's configuration, data and execution record.

    Attributes:
        run_id: Unique identifier for this run; the key every artifact and
            log line is eventually tied back to.
        dataset_path: Location of the dataset being processed.
        configuration: Normalized metadata — the only source of column
            identities anywhere in the engine.
        pipeline_config: Behavioural settings for this run.
        raw_dataset: The dataset exactly as loaded, never mutated, so any
            stage can compare against the original.
        prepared_dataset: The cleaned dataset produced by DataPreprocessor.
        groups: One entry per business key (or exactly one in single-series
            mode).
        series: The forecast-ready time series — this phase's final output
            and the Model Training Engine's direct input.
        frequency: Detected sampling grain, recorded but never normalized.
        current_key: Group being processed, set while iterating per key so
            failures can be attributed to a specific business key.
        metadata: Free-form run facts (row counts, drop counts) for logging.
        stages: Ordered execution record.
    """

    run_id: str
    dataset_path: Path
    configuration: ForecastConfiguration
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig.default)

    raw_dataset: pd.DataFrame | None = None
    prepared_dataset: pd.DataFrame | None = None
    groups: list["ForecastGroup"] = field(default_factory=list)
    series: list["ForecastSeries"] = field(default_factory=list)

    # Data Quality Assessment output (describes the raw dataset) and the
    # Preprocessing summary (records what was done to produce the curated
    # one). Together they reconcile raw row counts with curated ones.
    quality_report: "QualityReport | None" = None
    preprocessing_summary: "PreprocessingSummary | None" = None
    curated_dataset_uri: str | None = None
    # One record per forecast key describing whether that key's winning
    # fitted model was persisted, and where. Written by the Persist
    # Winning Models stage.
    model_storage_results: list[dict[str, Any]] = field(default_factory=list)
    # The run's exported forecast CSV — where it was written, or why not.
    # Written by the Export Forecasts stage.
    forecast_export_result: dict[str, Any] = field(default_factory=dict)
    # Business insights / LLM trace mirrored outside MLflow. Written by
    # the Mirror Artifacts stage.
    artifacts_mirror_result: dict[str, Any] = field(default_factory=dict)

    # Model Training output: one record per (group, model) pair, holding the
    # fitted estimator and its training metadata. This is the next phase's
    # direct input.
    training_report: "TrainingReport | None" = None
    selected_models: list[str] | None = None

    # Derived feature columns selected for this run (Priority C) —
    # `lag_*`/`rolling_mean_*`/calendar feature ids, consumed by the
    # tree-based models. `None` means this run never mentioned the field at
    # all (every run before this feature existed, or one that explicitly
    # accepts every default), never confused with an explicit empty
    # selection — see `forecast_engine.config.derived_features_config`.
    derived_features: list[str] | None = None

    # Model used when every evaluated model fails validation (Section 6.9).
    # Recorded on the context so the run summary, and MLflow through it,
    # report the fallback that was actually configured for this run.
    fallback_model: str | None = None

    # Evaluation output: backtest metrics, forward forecasts and the
    # elimination verdict per (group, model). Its surviving models are
    # the ranking phase's direct input.
    evaluation_report: "EvaluationReport | None" = None

    # Explainability output (Section 6.10): SHAP/feature-importance results
    # for every surviving model, generated before Model Ranking runs — one
    # of Ranking's composite inputs, and later the LLM Insight Engine's.
    explainability_report: "ExplainabilityReport | None" = None

    # Ranking output: composite score, backtesting rank and final rank per
    # surviving model, per group (Section 6.6). Final Production Model
    # Selection's direct input.
    ranking_report: "RankingReport | None" = None

    # Final Production Model Selection output (Section 6.9): the model
    # actually eligible to forecast for each group, after Drift Validation.
    # This is Phase 7B's deliverable and the LLM Insight Engine's / MLflow
    # Logging's direct input.
    production_selection_report: "ProductionSelectionReport | None" = None

    # LLM Insight Engine output (Section 6.12): business-readable
    # interpretation of everything above. Purely descriptive — nothing here
    # feeds back into any forecasting decision.
    insight_report: "BusinessInsightReport | None" = None
    # The detailed per-call LLM trace (Section 13.4) — one record per
    # attempt, including failed/retried ones. `insight_report.trace_summary`
    # carries only the aggregate; this is the debuggable detail behind it.
    llm_trace: dict[str, Any] = field(default_factory=dict)

    # MLflow tracking output (Section 6.13): whether this run's parameters,
    # metrics, artifacts and winner models were logged/registered. Purely
    # observational — a failure here never invalidates the forecast
    # results recorded everywhere else on this context.
    tracking_result: "TrackingResult | None" = None

    frequency: str | None = None
    current_key: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
    stages: list[StageRecord] = field(default_factory=list)

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    # Fired after every stage transition (begin/complete/fail) — how the
    # run's live progress reaches a caller *while it is still executing*,
    # rather than only once at the very end via `summary()`. Never
    # serialized: excluded from `repr`/`compare` and from `summary()`'s
    # hand-built dict, since a live-status writer is a delivery mechanism,
    # not part of the run's own record.
    on_stage_change: "Callable[[PipelineContext], None] | None" = field(default=None, repr=False, compare=False)

    # Start a new run context; generates a run_id when omitted
    @classmethod
    def create(
        cls,
        dataset_path: str | Path,
        configuration: ForecastConfiguration,
        pipeline_config: PipelineConfig | None = None,
        run_id: str | None = None,
    ) -> "PipelineContext":
        return cls(
            run_id=run_id or f"fe-run-{uuid.uuid4().hex[:12]}",
            dataset_path=Path(dataset_path),
            configuration=configuration,
            pipeline_config=pipeline_config or PipelineConfig.default(),
        )

    # Forecast mode for this run, derived from the metadata
    @property
    def mode(self) -> ForecastMode:
        return self.configuration.mode

    # Number of forecasting groups generated so far
    @property
    def group_count(self) -> int:
        return len(self.groups)

    # Number of forecast-ready series built so far
    @property
    def series_count(self) -> int:
        return len(self.series)

    # Record that a stage has started and return its record
    def begin_stage(self, name: str) -> StageRecord:
        record = StageRecord(name=name)
        self.stages.append(record)
        self._notify_stage_change()
        return record

    # Mark `record` finished, optionally with a one-line summary
    def complete_stage(self, record: StageRecord, detail: str | None = None) -> None:
        record.status = StageStatus.COMPLETED.value
        record.completed_at = datetime.now(timezone.utc)
        record.detail = detail
        self._notify_stage_change()

    # Mark `record` failed, capturing the error for the run trail
    def fail_stage(self, record: StageRecord, error: Exception) -> None:
        record.status = StageStatus.FAILED.value
        record.completed_at = datetime.now(timezone.utc)
        record.detail = str(error)
        self._notify_stage_change()

    # Invoke the live-status callback, if one is set; a broken listener must
    # never interrupt the forecasting run it is only reporting on
    def _notify_stage_change(self) -> None:
        if self.on_stage_change is None:
            return
        try:
            self.on_stage_change(self)
        except Exception:  # noqa: BLE001 - reporting progress must never fail the run
            pass

    # Close the run, stamping its completion time
    def finish(self) -> None:
        self.completed_at = datetime.now(timezone.utc)

    # Attach arbitrary run facts (row counts, drops) to the context
    def record(self, **facts: Any) -> None:
        self.metadata.update(facts)

    # Serializable summary of the run, excluding the DataFrames themselves
    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_path": str(self.dataset_path),
            "configuration": self.configuration.to_dict(),
            "frequency": self.frequency,
            "mode": self.mode.value,
            "group_count": self.group_count,
            "series_count": self.series_count,
            # Per-group descriptors plus a bounded history tail — what the
            # Results dashboard plots actuals from.
            "forecast_groups": [series.to_dict() for series in self.series],
            "quality_report": self.quality_report.to_dict() if self.quality_report else None,
            "preprocessing_summary": (
                self.preprocessing_summary.to_dict() if self.preprocessing_summary else None
            ),
            "curated_dataset_uri": self.curated_dataset_uri,
            "model_storage_results": self.model_storage_results,
            "forecast_export_result": self.forecast_export_result,
            "artifacts_mirror_result": self.artifacts_mirror_result,
            "selected_models": self.selected_models,
            "fallback_model": self.fallback_model,
            "derived_features": self.derived_features,
            "training_report": self.training_report.to_dict() if self.training_report else None,
            "evaluation_report": self.evaluation_report.to_dict() if self.evaluation_report else None,
            "explainability_report": (
                self.explainability_report.to_dict() if self.explainability_report else None
            ),
            "ranking_report": self.ranking_report.to_dict() if self.ranking_report else None,
            "production_selection_report": (
                self.production_selection_report.to_dict() if self.production_selection_report else None
            ),
            "insight_report": self.insight_report.to_dict() if self.insight_report else None,
            "llm_trace": self.llm_trace,
            "tracking_result": self.tracking_result.to_dict() if self.tracking_result else None,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "completed_at": self.completed_at.isoformat(timespec="seconds") if self.completed_at else None,
            "metadata": self.metadata,
            "stages": [stage.to_dict() for stage in self.stages],
        }
