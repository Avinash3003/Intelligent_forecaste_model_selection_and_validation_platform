"""The state object every stage reads from and writes back to.

Threading a growing argument list between stages would mean changing every
signature to add a field; instead each stage takes what it needs from here.
It also accumulates a timed record of every stage, which becomes the MLflow
run trail, and represents exactly one run.

Deliberately mutable — accumulating state is the point.
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
    from forecast_engine.parallel.ray_executor import StagedKeyExecution
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
    """One stage's timed record, so a run can say where its time went and
    which stage failed."""

    name: str
    status: str = StageStatus.RUNNING.value
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    detail: str | None = None
    # What the stage's own work actually took, when the stage can measure
    # it better than the driver clock can.
    #
    # Ray runs train -> evaluate -> explain -> rank -> select for one key
    # inside a single task, so every one of those phases has already
    # finished by the time the driver opens its stage. Driver elapsed time
    # then reads ~0s for four real stages and attributes the whole parallel
    # window to Train Models. Measuring 0.0s is not evidence that nothing
    # happened, and the UI must not imply that it is.
    measured_seconds: float | None = None
    # Set only for a stage that fanned real Ray tasks out across keys —
    # total/completed/failed/running counts plus per-task durations. None
    # for every stage that never had independent parallel units to report.
    parallel_tasks: dict[str, Any] | None = None

    # What the stage's work took: its own measurement when it has one,
    # otherwise the driver's wall clock.
    @property
    def duration_seconds(self) -> float | None:
        if self.measured_seconds is not None:
            return self.measured_seconds
        return self._elapsed_seconds

    # Driver-side orchestration time, kept separate from the logical
    # duration above so the two are never confused.
    @property
    def _elapsed_seconds(self) -> float | None:
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
            "measured_seconds": self.measured_seconds,
            "orchestration_seconds": self._elapsed_seconds,
            "detail": self.detail,
            "parallel_tasks": self.parallel_tasks,
        }


@dataclass
class PipelineContext:
    """One run's configuration, data and execution record.

    run_id ties every artifact and log line back to this run. configuration
    is the only source of column identities in the engine. raw_dataset is
    never mutated, so any stage can compare against the original;
    prepared_dataset is the cleaned version. groups holds one entry per
    business key, series the forecast-ready output. current_key is set while
    iterating so a failure can be attributed to a specific key.
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

    # Key-parallel execution state, shared across Train/Evaluate/Explain/
    # Rank & Select — each of those four stages is its own real Ray
    # fan-out through this one executor, not a single cached result.
    # Set only when the run fans out per key.
    key_stage_executor: "StagedKeyExecution | None" = None

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
    def complete_stage(
        self,
        record: StageRecord,
        detail: str | None = None,
        measured_seconds: float | None = None,
        parallel_tasks: dict[str, Any] | None = None,
    ) -> None:
        record.status = StageStatus.COMPLETED.value
        record.completed_at = datetime.now(timezone.utc)
        record.detail = detail
        if measured_seconds is not None:
            record.measured_seconds = measured_seconds
        if parallel_tasks is not None:
            record.parallel_tasks = parallel_tasks
        self._notify_stage_change()

    # Mark `record` failed, capturing the error for the run trail
    def fail_stage(self, record: StageRecord, error: Exception) -> None:
        record.status = StageStatus.FAILED.value
        record.completed_at = datetime.now(timezone.utc)
        record.detail = str(error)
        self._notify_stage_change()

    # Update a still-RUNNING stage's parallel-task counts, without closing
    # it — lets a live poll see tasks completing one by one, not just the
    # final total once the whole stage is done.
    def update_stage_progress(self, record: StageRecord, parallel_tasks: dict[str, Any]) -> None:
        record.parallel_tasks = parallel_tasks
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
