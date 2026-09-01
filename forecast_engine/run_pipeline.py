"""The engine's entry point: runs every stage in order.

    load -> detect frequency -> assess quality -> preprocess -> build series
    -> train -> evaluate -> explain -> rank & select -> persist -> export
    -> insights -> mirror -> track

This file owns sequencing and error handling only; how each step behaves
lives in its own stage class and config. Adding a stage means appending
here, not restructuring.

Run locally:

    python -m forecast_engine.run_pipeline \\
        --dataset store_item_dev.csv \\
        --date-column date --target-column sales \\
        --key-columns store item
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from forecast_engine.config.drift_config import DriftValidationConfig
from forecast_engine.config.evaluation_config import EvaluationConfig
from forecast_engine.config.explainability_config import ExplainabilityConfig
from forecast_engine.config.llm_config import LLMConfig
from forecast_engine.config.mlflow_config import MLflowConfig
from forecast_engine.config.derived_features_config import apply_to_model_config
from forecast_engine.config.model_config import ModelConfig
from forecast_engine.config.pipeline_config import PipelineConfig
from forecast_engine.config.ranking_config import RankingConfig
from forecast_engine.core.databricks_secrets import apply_azure_openai_cli_overrides
from forecast_engine.core.forecast_configuration import AggregationMethod, ForecastConfiguration
from forecast_engine.core import checkpoint, storage
from forecast_engine.core.live_status import LiveStatusWriter
from forecast_engine.core.pipeline_context import PipelineContext, StageStatus
from forecast_engine.core.pipeline_result import PipelineResultBuilder
from forecast_engine.parallel.key_workflow import KeyWorkflowConfig
from forecast_engine.parallel.ray_executor import StagedKeyExecution
from forecast_engine.s01_preprocessing.data_preprocessor import DataPreprocessor
from forecast_engine.s01_preprocessing.dataset_loader import DatasetLoader
from forecast_engine.s01_preprocessing.frequency_detector import FrequencyDetector
from forecast_engine.s01_preprocessing.group_generator import GroupGenerator
from forecast_engine.s01_preprocessing.series_builder import SeriesBuilder
from forecast_engine.s02_quality.quality_assessor import DataQualityAssessor
from forecast_engine.s03_storage.curated_writer import CuratedDatasetWriter, read_curated_dataset
from forecast_engine.s03_storage.model_writer import WinningModelWriter
from forecast_engine.s03_storage.forecast_export_writer import ForecastExportWriter
from forecast_engine.s03_storage.artifacts_mirror_writer import ArtifactsMirrorWriter
from forecast_engine.s03_storage.volume_sync import sync_outputs_to_volume
from forecast_engine.s04_training.curated_validator import CuratedDatasetValidator
from forecast_engine.s04_training.model_trainer import ModelTrainer
from forecast_engine.s05_models.model_registry import ModelRegistry
from forecast_engine.s06_evaluation.evaluation_pipeline import EvaluationPipeline
from forecast_engine.s07_explainability.explainability_pipeline import ExplainabilityPipeline
from forecast_engine.s10_selection.production_pipeline import ProductionSelectionPipeline
from forecast_engine.s11_llm.insight_engine import LLMInsightEngine
from forecast_engine.s12_tracking.tracking_pipeline import MLflowTrackingPipeline
from forecast_engine.utils.exceptions import ConfigurationError, DataQualityError, ForecastEngineError

# Platform-wide bound on the forward forecast horizon (Section 3: minimum
# 12-month horizon, extended by the platform down to 6 and up to 60).
MIN_FORECAST_HORIZON = 6
MAX_FORECAST_HORIZON = 60

# The seven Databricks task boundaries, each a tuple of the private stage
# methods below it runs in order. Mirrors backend/app/services/
# pipeline_stages.py's PIPELINE_PHASES exactly (kept in sync by hand, same
# convention as PIPELINE_STAGES — see the NAMING CONTRACT note on the class)
# since the two packages never import each other.
PHASE_STAGE_METHODS: dict[str, tuple[str, ...]] = {
    "load_prepare": (
        "_load_dataset",
        "_detect_frequency",
        "_assess_quality",
        "_preprocess",
        "_persist_curated_dataset",
        "_verify_curated_dataset",
    ),
    "build_series": ("_generate_groups", "_build_series"),
    "train_models": ("_train_models",),
    "evaluate_models": ("_evaluate_models",),
    "explain_models": ("_generate_explainability",),
    "rank_select": ("_select_production_models",),
    "publish_results": (
        "_persist_winning_models",
        "_export_forecasts",
        "_generate_business_insights",
        "_mirror_artifacts",
        "_track_to_mlflow",
    ),
}
PHASE_ORDER: tuple[str, ...] = tuple(PHASE_STAGE_METHODS)
FIRST_PHASE = PHASE_ORDER[0]
LAST_PHASE = PHASE_ORDER[-1]
# Phases that resume the Ray key-execution state a prior task built —
# Train is the first phase to touch it (builds fresh) and Publish never
# touches it at all, so neither belongs here.
_PHASES_RESUMING_KEY_EXECUTION = frozenset({"evaluate_models", "explain_models", "rank_select"})


class ForecastEnginePipeline:
    """Runs the engine end to end.

    Stages are injected rather than built inline, so a deployment or test can
    substitute one without touching the orchestration.
    """

    # NAMING CONTRACT — one vocabulary across engine and UI. Each
    # `begin_stage(...)` label below (what the UI's stage trail shows) is
    # deliberately short and Title Case, so it reads uniformly in a
    # seventeen-row trail. Backend mirror: `backend/app/services/
    # deployment_service.py`'s PIPELINE_STAGES (kept in sync by
    # tests/backend/test_stage_trail.py).

    def __init__(
        self,
        pipeline_config: PipelineConfig | None = None,
        loader: DatasetLoader | None = None,
        frequency_detector: FrequencyDetector | None = None,
        quality_assessor: DataQualityAssessor | None = None,
        preprocessor: DataPreprocessor | None = None,
        curated_writer: CuratedDatasetWriter | None = None,
        model_writer: WinningModelWriter | None = None,
        forecast_export_writer: ForecastExportWriter | None = None,
        artifacts_mirror_writer: ArtifactsMirrorWriter | None = None,
        group_generator: GroupGenerator | None = None,
        series_builder: SeriesBuilder | None = None,
        model_config: ModelConfig | None = None,
        curated_validator: CuratedDatasetValidator | None = None,
        trainer: ModelTrainer | None = None,
        evaluation_config: EvaluationConfig | None = None,
        evaluator: EvaluationPipeline | None = None,
        ranking_config: RankingConfig | None = None,
        drift_config: DriftValidationConfig | None = None,
        production_selector: ProductionSelectionPipeline | None = None,
        explainability_config: ExplainabilityConfig | None = None,
        explainability_generator: ExplainabilityPipeline | None = None,
        llm_config: LLMConfig | None = None,
        insight_engine: LLMInsightEngine | None = None,
        mlflow_config: MLflowConfig | None = None,
        tracking_pipeline: MLflowTrackingPipeline | None = None,
        parallel_keys: bool = False,
    ) -> None:
        # Off by default: an unflagged run executes exactly as it always has.
        self._parallel_keys = parallel_keys
        self._config = pipeline_config or PipelineConfig.default()
        self._model_config = model_config or ModelConfig.default()
        self._loader = loader or DatasetLoader()
        self._frequency_detector = frequency_detector or FrequencyDetector()
        self._quality_assessor = quality_assessor or DataQualityAssessor(self._config.quality)
        self._preprocessor = preprocessor or DataPreprocessor(
            self._config.preprocessing,
            self._config.conversion,
            self._config.quality,
            self._config.aggregation,
        )
        self._curated_writer = curated_writer or CuratedDatasetWriter(self._config.curated_storage)
        self._model_writer = model_writer or WinningModelWriter(self._config.model_storage)
        self._forecast_export_writer = forecast_export_writer or ForecastExportWriter(self._config.forecast_export)
        self._artifacts_mirror_writer = artifacts_mirror_writer or ArtifactsMirrorWriter(self._config.artifacts_mirror)
        self._group_generator = group_generator or GroupGenerator(self._config.grouping)
        self._series_builder = series_builder or SeriesBuilder(self._config.grouping)
        self._curated_validator = curated_validator or CuratedDatasetValidator(self._config.quality)
        self._trainer = trainer or ModelTrainer(self._model_config)
        self._evaluation_config = evaluation_config or EvaluationConfig.default()
        self._evaluator = evaluator or EvaluationPipeline(
            ModelRegistry(self._model_config), self._evaluation_config
        )
        self._explainability_config = explainability_config or ExplainabilityConfig.default()
        self._explainability_generator = explainability_generator or ExplainabilityPipeline(
            ModelRegistry(self._model_config), self._explainability_config
        )
        self._ranking_config = ranking_config or RankingConfig.default()
        self._drift_config = drift_config or DriftValidationConfig.default()
        self._production_selector = production_selector or ProductionSelectionPipeline(
            ModelRegistry(self._model_config),
            self._ranking_config,
            self._drift_config,
            self._model_config,
            forecast_horizon=self._evaluation_config.forecast_horizon,
        )
        self._llm_config = llm_config or LLMConfig.default()
        self._insight_engine = insight_engine or LLMInsightEngine(self._llm_config)
        self._mlflow_config = mlflow_config or MLflowConfig.default()
        self._tracking_pipeline = tracking_pipeline or MLflowTrackingPipeline(self._mlflow_config)

    def run(
        self,
        dataset_path: str | Path,
        configuration: ForecastConfiguration,
        run_id: str | None = None,
        selected_models: list[str] | None = None,
        fallback_model: str | None = None,
        dataset_name: str | None = None,
        live_status_path: str | Path | None = None,
        started_by_user_id: str | None = None,
        started_by_display_name: str | None = None,
        derived_features: list[str] | None = None,
    ) -> PipelineContext:
        """Run every stage for one dataset, in one process.

        Stages share one PipelineContext, which each annotates as it
        completes. That context is returned whether the run succeeded or died
        partway through.

        Failures are handled twice on purpose: the stage records its own
        failure, then the error reaches here, closes the MLflow run as FAILED
        and re-raises. So a run that dies in preprocessing still leaves a
        complete, queryable record instead of looking like an empty success.

        run_id is reused when the caller already minted one, so the backend's
        job id and the engine's agree. selected_models defaults to every
        registered model. live_status_path, when set, receives the stage trail
        after every transition so a poller can show live progress.
        started_by_* is recorded on the MLflow run, so who started it survives
        even a run that fails or is cancelled.

        Raises ForecastEngineError after recording the failure.
        """
        # On failure, the failing stage is
        # recorded on the context before the error propagates, so the
        # partial run remains auditable.

        # A user-chosen fallback overrides the configured default before any
        # stage runs, so every stage that consults ModelConfig — including
        # Final Production Model Selection and MLflow logging — sees the
        # same model without needing it threaded through separately.
        if fallback_model:
            self._apply_fallback_model(fallback_model)

        context = PipelineContext.create(
            dataset_path=dataset_path,
            configuration=configuration,
            pipeline_config=self._config,
            run_id=run_id,
        )
        context.selected_models = selected_models
        context.fallback_model = self._model_config.fallback_model
        context.derived_features = derived_features

        # Set before stage 1 so even "Load Dataset" begins reporting live,
        # rather than only every stage after the first.
        if live_status_path is not None:
            context.on_stage_change = LiveStatusWriter(live_status_path)

        # The MLflow Parent Run is opened *before* the first stage so that a
        # run dying in preprocessing still leaves a real, FAILED MLflow run
        # carrying the reason. Tracking remains purely observational: the
        # engine never reads back from MLflow, and a tracking failure here
        # only downgrades what gets recorded, never what gets forecast.
        # The caller's display name wins over the file's own: a staged
        # upload is named "{file_id}_{original}.csv" on disk, which is not
        # what a user recognises in a run-history view.
        context.tracking_result = self._tracking_pipeline.begin(
            context.run_id,
            dataset_name or Path(dataset_path).name,
            started_by_user_id=started_by_user_id,
            started_by_display_name=started_by_display_name,
        )

        for phase in PHASE_ORDER:
            self.run_phase(phase, context)
        return context

    # Run one phase's stages against an existing context — the shared
    # sequencing every entry point (run() above, and the Databricks
    # multi-task path below) actually executes. A phase failure records the
    # failed stage on MLflow and re-raises; the caller's context still holds
    # everything completed before it.
    def run_phase(self, phase: str, context: PipelineContext) -> PipelineContext:
        methods = PHASE_STAGE_METHODS.get(phase)
        if methods is None:
            raise ConfigurationError(f"Unknown pipeline phase '{phase}'.")

        try:
            for method_name in methods:
                getattr(self, method_name)(context)
        except Exception as exc:
            context.tracking_result = self._tracking_pipeline.fail(context.run_id, exc, _failed_stage(context))
            context.finish()
            raise

        if phase == LAST_PHASE:
            context.finish()
        return context

    # One Databricks task's slice of a run: start fresh (first phase) or
    # resume from the previous task's checkpoint, run exactly one phase,
    # then checkpoint the result — success or failure — so the next task (or
    # a retry of this one) has real persisted state to resume from.
    def run_checkpointed_stage(
        self,
        stage: str,
        run_id: str,
        dataset_path: str | Path,
        configuration: ForecastConfiguration,
        artifacts_root: str,
        selected_models: list[str] | None = None,
        fallback_model: str | None = None,
        dataset_name: str | None = None,
        live_status_path: str | Path | None = None,
        started_by_user_id: str | None = None,
        started_by_display_name: str | None = None,
        derived_features: list[str] | None = None,
    ) -> PipelineContext:
        if fallback_model:
            self._apply_fallback_model(fallback_model)

        if stage == FIRST_PHASE:
            context = PipelineContext.create(
                dataset_path=dataset_path,
                configuration=configuration,
                pipeline_config=self._config,
                run_id=run_id,
            )
            context.selected_models = selected_models
            context.fallback_model = self._model_config.fallback_model
            context.derived_features = derived_features
            context.tracking_result = self._tracking_pipeline.begin(
                context.run_id,
                dataset_name or Path(dataset_path).name,
                started_by_user_id=started_by_user_id,
                started_by_display_name=started_by_display_name,
            )
        else:
            context, snapshot = checkpoint.load(artifacts_root, run_id)
            if snapshot is not None and stage in _PHASES_RESUMING_KEY_EXECUTION:
                context.key_stage_executor = StagedKeyExecution.resume(self._key_workflow_config(context), snapshot)
            # A fresh process has no MLflow run open in its own fluent
            # state — only begin()'s task opened one. Without this,
            # every later task's track()/fail() reports "no run was open".
            if context.tracking_result is not None:
                self._tracking_pipeline.resume(context.tracking_result.run_id)

        if live_status_path is not None:
            context.on_stage_change = LiveStatusWriter(live_status_path)

        try:
            self.run_phase(stage, context)
        finally:
            checkpoint.save(context, artifacts_root)

        return context

    # Override the configured fallback model for this run
    def _apply_fallback_model(self, fallback_model: str) -> None:
        # ModelConfig is frozen, so a replacement is built and the one stage
        # that already captured it (production selection) is rebuilt against
        # the new config — otherwise it would keep validating against the
        # default fallback.
        self._model_config = replace(self._model_config, fallback_model=fallback_model)
        self._production_selector = ProductionSelectionPipeline(
            ModelRegistry(self._model_config),
            self._ranking_config,
            self._drift_config,
            self._model_config,
            forecast_horizon=self._evaluation_config.forecast_horizon,
        )

    # Stage 1 — read the file into memory, unmodified
    def _load_dataset(self, context: PipelineContext) -> None:
        record = context.begin_stage("Load Dataset")
        try:
            context.raw_dataset = self._loader.load(context.dataset_path)
            rows, columns = context.raw_dataset.shape
            context.record(raw_rows=rows, raw_columns=columns)
            context.complete_stage(record, f"Loaded {rows:,} rows × {columns} columns.")
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Stage 2 — infer the sampling grain
    def _detect_frequency(self, context: PipelineContext) -> None:
        record = context.begin_stage("Detect Frequency")
        try:
            # Runs against the raw dataset, before cleaning, so the detected
            # grain reflects the data as the user supplied it.
            configuration = context.configuration
            configuration.validate_against_columns(list(context.raw_dataset.columns))

            context.frequency = self._frequency_detector.detect(
                context.raw_dataset[configuration.date_column]
            )
            context.complete_stage(record, f"Detected frequency: {context.frequency}.")
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Stage 3 — analyse the raw dataset without modifying it
    def _assess_quality(self, context: PipelineContext) -> None:
        record = context.begin_stage("Assess Quality")
        try:
            # Runs before preprocessing so the report describes the data
            # exactly as uploaded, and gates it: a dataset judged unsuitable
            # is rejected here rather than being cleaned into something
            # misleading.
            report = self._quality_assessor.assess(
                context.raw_dataset, context.configuration, context.frequency
            )
            context.quality_report = report

            if not report.is_forecastable:
                raise DataQualityError(
                    "Dataset unsuitable for forecasting: " + " ".join(report.suitability_reasons)
                )

            context.complete_stage(
                record,
                f"Suitability: {report.suitability.value}; "
                f"{report.duplicate_rows:,} duplicate row(s), "
                f"{report.total_observations:,} usable observation(s).",
            )
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Stage 4 — produce the curated dataset (only stage that modifies data)
    def _preprocess(self, context: PipelineContext) -> None:
        # The raw dataset stays on the context untouched for comparison.
        record = context.begin_stage("Preprocess Dataset")
        try:
            curated, summary = self._preprocessor.prepare(
                context.raw_dataset, context.configuration, context.frequency
            )
            summary.detected_frequency = context.frequency

            context.prepared_dataset = curated
            context.preprocessing_summary = summary
            context.record(**summary.to_dict())

            context.complete_stage(
                record,
                f"{summary.rows_read:,} rows in, {summary.curated_rows:,} curated "
                f"({summary.rows_removed:,} removed).",
            )
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Stage 5 — write the curated dataset to its own location
    def _persist_curated_dataset(self, context: PipelineContext) -> None:
        if not self._config.curated_storage.enabled:
            return

        # The uploaded file is never touched; curated output is filed under
        # the run id so it is always traceable to the run that produced it.
        record = context.begin_stage("Persist Curated")
        try:
            uri = self._curated_writer.write(
                context.prepared_dataset, context.run_id, context.dataset_path.name
            )
            context.curated_dataset_uri = uri
            if context.preprocessing_summary:
                context.preprocessing_summary.curated_dataset_uri = uri

            context.complete_stage(record, f"Curated dataset written to {uri}.")
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Stage 6 — internal integrity check before training
    def _verify_curated_dataset(self, context: PipelineContext) -> None:
        # Not user-facing and produces no report: it exists to fail fast if a
        # curated dataset ever reaches training in a state preprocessing
        # should have prevented.
        record = context.begin_stage("Verify Curated")
        try:
            self._curated_validator.validate(context.prepared_dataset, context.configuration)
            context.complete_stage(record, "Curated dataset integrity verified.")
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Stage 7 — split the curated dataset into per-business-key groups
    def _generate_groups(self, context: PipelineContext) -> None:
        record = context.begin_stage("Generate Groups")
        try:
            # A Databricks task resuming from a checkpoint never carries the
            # prepared DataFrame in memory (see checkpoint.save) — read it
            # back from where Load & Prepare persisted it instead.
            if context.prepared_dataset is None:
                if not context.curated_dataset_uri:
                    raise ConfigurationError(
                        "Build Series requires a curated dataset, but none was persisted for this run."
                    )
                context.prepared_dataset = read_curated_dataset(
                    context.curated_dataset_uri, context.configuration.date_column
                )

            context.groups = self._group_generator.generate(
                context.prepared_dataset, context.configuration
            )
            context.record(group_count=context.group_count)
            context.complete_stage(
                record, f"{context.group_count:,} group(s) in {context.mode.value} mode."
            )
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Stage 8 — project each group into a forecast-ready series
    def _build_series(self, context: PipelineContext) -> None:
        record = context.begin_stage("Build Series")
        try:
            context.series = self._series_builder.build(
                context.groups, context.configuration, context.frequency
            )

            # Surface how many keys fall short of the history the accuracy
            # target assumes, so the caller can flag them (Section 10).
            below_minimum = sum(1 for series in context.series if not series.meets_minimum_history)
            context.record(series_count=context.series_count, series_below_minimum_history=below_minimum)
            context.complete_stage(
                record,
                f"{context.series_count:,} series built; {below_minimum:,} below minimum history.",
            )
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # The immutable config every key-parallel stage shares — built from
    # this pipeline's own collaborators plus what the run itself chose.
    def _key_workflow_config(self, context: PipelineContext) -> KeyWorkflowConfig:
        return KeyWorkflowConfig(
            model=self._model_config,
            evaluation=self._evaluation_config,
            explainability=self._explainability_config,
            ranking=self._ranking_config,
            drift=self._drift_config,
            selected_models=(None if context.selected_models is None else tuple(context.selected_models)),
        )

    # One executor, shared by all four key-parallel stages below — built
    # once so Evaluate can see Train's real per-key output, not a copy.
    def _stage_executor(self, context: PipelineContext) -> StagedKeyExecution:
        if context.key_stage_executor is None:
            context.key_stage_executor = StagedKeyExecution(context.series, self._key_workflow_config(context))
        return context.key_stage_executor

    # Live task-by-task progress for one stage, while it is still running —
    # not just its final count once every key has finished.
    def _progress(self, context: PipelineContext, record):
        def on_progress(_stage_name: str, telemetry: dict[str, Any]) -> None:
            context.update_stage_progress(record, telemetry)

        return on_progress

    # Stage 9 — train every selected model on every forecasting group.
    # A genuine Ray fan-out across every key when parallel; every task asks
    # for one CPU and completes independently before this stage closes.
    def _train_models(self, context: PipelineContext) -> None:
        record = context.begin_stage("Train Models")
        try:
            # Individual training failures are captured in the report rather
            # than raised, so one bad key or model never ends the run. Only
            # a configuration fault (unknown model, broken adapter) propagates.
            telemetry = None
            if self._parallel_keys:
                report, telemetry = self._stage_executor(context).run_training(self._progress(context, record))
            else:
                report = self._trainer.train_all(context.series, context.selected_models)
            context.training_report = report

            context.record(
                models_trained=report.trained_count,
                models_failed=report.failed_count,
                models_skipped=report.skipped_count,
                models_unavailable=report.unavailable_count,
            )
            context.complete_stage(
                record,
                f"{report.trained_count:,} trained, {report.failed_count:,} failed, "
                f"{report.skipped_count:,} skipped, {report.unavailable_count:,} unavailable "
                f"across {report.groups_trained:,} group(s).",
                measured_seconds=telemetry["wall_seconds"] if telemetry else report.duration_seconds,
                parallel_tasks=telemetry,
            )
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Stage 10 — backtest, forecast forward and eliminate. A genuine Ray
    # fan-out of its own, depending only on Train's real per-key output.
    def _evaluate_models(self, context: PipelineContext) -> None:
        record = context.begin_stage("Evaluate Models")
        try:
            # Produces the surviving-model set that ranking consumes.
            # Individual (group, model) failures are captured in the report
            # rather than raised, so one bad pair never ends the run.
            telemetry = None
            if self._parallel_keys:
                report, telemetry = self._stage_executor(context).run_evaluation(self._progress(context, record))
            else:
                trained = context.training_report.trained_models() if context.training_report else []
                report = self._evaluator.evaluate_all(context.series, trained)
            context.evaluation_report = report

            context.record(
                models_survived=report.survived_count,
                models_eliminated=report.eliminated_count,
                models_evaluation_failed=report.failed_count,
            )
            # This stage's own real Ray fan-out, timed at its own
            # orchestration boundary — not the driver clock of a different
            # stage, and not another stage's task durations reused here.
            context.complete_stage(
                record,
                measured_seconds=telemetry["wall_seconds"] if telemetry else report.duration_seconds,
                parallel_tasks=telemetry,
                detail=f"{report.survived_count:,} survived, {report.eliminated_count:,} eliminated, "
                f"{report.failed_count:,} failed across {report.groups_evaluated:,} group(s). "
                f"{report.model_fit_count:,} model fit(s) "
                f"({report.backtest_windows_evaluated:,} backtest fold(s) "
                f"+ {report.forecasts_refit:,} forward-forecast refit(s), "
                f"{report.forecasts_reused:,} forward forecast(s) reused training's own fit) — "
                f"backtest {report.backtest_seconds:.1f}s, forecast {report.forecast_generation_seconds:.1f}s, "
                f"validation {report.validation_seconds:.1f}s.",
            )
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Stage 11 — SHAP / feature importance generation (Section 6.10). A
    # genuine Ray fan-out depending only on Evaluate's real per-key output.
    def _generate_explainability(self, context: PipelineContext) -> None:
        # Runs on every surviving (group, model) pair, strictly before Model
        # Ranking — its output is one of Ranking's composite inputs. Per-pair
        # failures are captured on the report by the engine itself, never
        # raised, so one model's explainability computation cannot block the
        # rest of its group.
        record = context.begin_stage("Explain Models")
        try:
            telemetry = None
            if self._parallel_keys:
                report, telemetry = self._stage_executor(context).run_explainability(self._progress(context, record))
            else:
                trained = context.training_report.trained_models() if context.training_report else []
                report = self._explainability_generator.generate_all(context.evaluation_report, trained, context.series)
            context.explainability_report = report

            context.record(explainability_results=len(report.results))
            # This stage's own real Ray fan-out — see Evaluate Models above.
            context.complete_stage(
                record,
                measured_seconds=telemetry["wall_seconds"] if telemetry else report.duration_seconds,
                parallel_tasks=telemetry,
                detail=f"Explainability generated for {len(report.results):,} surviving model(s).",
            )
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Stage 12 — rank survivors and select each group's production model.
    # A genuine Ray fan-out depending only on Explain's real per-key output.
    def _select_production_models(self, context: PipelineContext) -> None:
        record = context.begin_stage("Rank & Select")
        try:
            # Ranking consumes Stage 11's ExplainabilityReport rather than
            # computing SHAP itself. Ranking and Final Production Model
            # Selection each isolate failure at their own grain (per group);
            # a configuration-level fault is the only thing that reaches
            # this try/except.
            telemetry = None
            if self._parallel_keys:
                ranking_report, selection_report, telemetry = self._stage_executor(context).run_rank_select(
                    self._progress(context, record)
                )
            else:
                ranking_report, selection_report = self._production_selector.run(
                    context.evaluation_report, context.explainability_report, context.series
                )
            context.ranking_report = ranking_report
            context.production_selection_report = selection_report

            context.record(
                models_selected=selection_report.selected_count,
                models_fallback_used=selection_report.fallback_count,
                groups_with_no_model_available=selection_report.unavailable_count,
            )
            # "Rank & Select" is one UI stage over two reports (ranking,
            # then final selection); their measured durations are summed so
            # the displayed time covers both rather than only the second —
            # this stage's own real Ray fan-out when parallel.
            ranking_seconds = getattr(ranking_report, "duration_seconds", None) or 0.0
            selection_seconds = getattr(selection_report, "duration_seconds", None) or 0.0
            context.complete_stage(
                record,
                measured_seconds=telemetry["wall_seconds"] if telemetry else ranking_seconds + selection_seconds,
                parallel_tasks=telemetry,
                detail=f"{selection_report.selected_count:,} selected, {selection_report.fallback_count:,} used "
                f"the fallback model, {selection_report.unavailable_count:,} had no model available "
                f"across {len(selection_report.results):,} group(s).",
            )
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Persist the winning fitted model for each forecast key
    def _persist_winning_models(self, context: PipelineContext) -> None:
        record = context.begin_stage("Persist Models")
        try:
            # Runs after selection so "the winner" is already decided, and
            # reads the fitted wrapper the training stage kept on its own
            # record — nothing is retrained here, and no candidate that lost
            # is written. A key that cannot be persisted is reported on its
            # own record rather than ending the run: the forecast itself is
            # already complete and correct by this point.
            selection = context.production_selection_report
            winners = list(selection.results) if selection else []
            trained = context.training_report.trained_models() if context.training_report else []

            results = self._model_writer.write_all(winners, trained, context.run_id)
            context.model_storage_results = results

            persisted = sum(1 for item in results if item["persisted"])
            context.record(models_persisted=persisted)
            context.complete_stage(
                record, f"{persisted:,} winning model(s) persisted of {len(results):,} group(s)."
            )
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Export the run's forecast output as one downloadable CSV
    def _export_forecasts(self, context: PipelineContext) -> None:
        record = context.begin_stage("Export Forecasts")
        selection = context.production_selection_report
        winners = list(selection.results) if selection else []

        result = self._forecast_export_writer.write(winners, context.run_id)
        context.forecast_export_result = result

        if result["persisted"]:
            context.complete_stage(record, f"{result['rows']:,} forecast row(s) exported across {len(winners):,} group(s).")
        else:
            context.complete_stage(record, result["error"] or "Nothing to export.")

    # Stage 13 — LLM business insights (Section 6.12)
    def _generate_business_insights(self, context: PipelineContext) -> None:
        record = context.begin_stage("Business Insights")
        try:
            # Purely descriptive: the LLM consumes the finished
            # PipelineResult and produces business-readable narrative. It
            # never touches training code and never influences any
            # forecasting decision, all of which are already final. A
            # misconfigured or unreachable provider degrades to a
            # clearly-marked "not generated" report rather than failing.
            pipeline_result = PipelineResultBuilder().build(context)
            report = self._insight_engine.generate(pipeline_result)
            context.insight_report = report
            if self._insight_engine.trace_store is not None:
                context.llm_trace = self._insight_engine.trace_store.to_dict()

            trace = report.trace_summary or {}
            detail = (
                f"Business insights {'generated' if report.available else 'skipped'}: {report.status}. "
                f"{trace.get('call_count', 0):,} LLM call(s), {trace.get('total_tokens', 0):,} token(s), "
                f"{trace.get('retry_count', 0):,} retr(y/ies)"
            )
            if trace.get("groundedness_rate") is not None:
                detail += f", groundedness {trace['groundedness_rate']:.0%}"
            if trace.get("estimated_cost_usd") is not None:
                detail += f", est. cost ${trace['estimated_cost_usd']:.4f}"
            detail += "."
            context.complete_stage(record, detail)
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Mirror business insights and the LLM trace outside MLflow
    def _mirror_artifacts(self, context: PipelineContext) -> None:
        record = context.begin_stage("Mirror Artifacts")
        result = self._artifacts_mirror_writer.write(
            context.insight_report.to_dict() if context.insight_report else {},
            context.llm_trace,
            context.run_id,
        )
        context.artifacts_mirror_result = result

        persisted = [p for p in result.get("persisted", []) if p["persisted"]]
        context.complete_stage(record, f"{len(persisted):,} artifact file(s) mirrored.")

    # Stage 14 — MLflow experiment tracking & model registry (Section 6.13)
    def _track_to_mlflow(self, context: PipelineContext) -> None:
        # The pipeline's last stage, run after Business Insights so the LLM
        # Business Summary artifact is available to log. Rebuilds
        # PipelineResult (cheap and pure — the same builder Stage 13 used)
        # rather than threading Stage 13's instance through, so this object
        # reflects everything produced up to this point, including the
        # insight report.
        record = context.begin_stage("MLflow Tracking")
        pipeline_result = PipelineResultBuilder().build(context)

        # The stage stays *open* across the tracking call, so anything
        # polling the live-status file sees "MLflow Tracking — Running"
        # instead of a finished 14-stage trail while artifact logging is
        # still going. That logging is not instant (plots, per-group model
        # registration), and previously the UI showed a run at 100% with no
        # current stage for the whole of it, which reads as a hang.
        #
        # The consolidated artifact must still record a *finished* trail
        # though — it is written during this very call, so the live object
        # would embed itself as running. `_summary_with_stage_completed`
        # resolves that: live status reports the truth, the artifact records
        # the outcome.
        summary = _summary_with_stage_completed(context.summary(), record.name)
        result = self._tracking_pipeline.track(pipeline_result, summary=summary)
        context.tracking_result = result
        context.complete_stage(record, "MLflow tracking complete.")

        # A tracking failure is recorded on context.tracking_result and
        # never raised: the forecasting run this stage is recording is
        # already complete and its results already available regardless of
        # whether MLflow could log them.
        record.detail = (
            f"MLflow tracking {result.status}"
            + (f" (run {result.run_id}, {result.models_registered} model(s) registered)." if result.logged else ".")
        )


def _summary_with_stage_completed(summary: dict[str, Any], stage_name: str) -> dict[str, Any]:
    """A copy of the summary with this stage marked Completed.

    Only the copy given to MLflow changes; the live context finishes
    honestly after tracking returns.
    """
    stages = []
    for stage in summary.get("stages") or []:
        if stage.get("name") == stage_name and stage.get("status") != StageStatus.COMPLETED.value:
            stage = {**stage, "status": StageStatus.COMPLETED.value, "detail": "MLflow tracking complete."}
        stages.append(stage)
    return {**summary, "stages": stages}


# Name of the stage a run died in, read from its own recorded trail
def _failed_stage(context: PipelineContext) -> str | None:
    for stage in reversed(context.stages):
        if stage.status == StageStatus.FAILED.value:
            return stage.name
    # No stage recorded a failure, so the error came from between stages
    # (or before the first one) — the last stage touched is the best
    # available locator.
    return context.stages[-1].name if context.stages else None


# Read a run's JSON configuration file, or {} when none was given
def load_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    if not args.config:
        return {}
    # Read through the adapter: on a DCS container the configuration is in
    # a UC Volume with no POSIX mount, so Path.read_text() cannot see it.
    payload = json.loads(storage.read_text(args.config))
    return payload if isinstance(payload, dict) else {}


# Apply the run-level keys a config file may carry, without overriding flags
def apply_config_run_options(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    """Let --config carry run_id, dataset_name, models, fallback_model and
    horizon as well as the column mapping.

    Exists for cloud runs: a wheel task's argument list is fixed in the
    bundle, so per-run values would arrive as always-passed job parameters,
    which argparse cannot survive when empty. It is also the only way to
    express multi-valued key/feature columns.

    An explicit CLI flag always wins, so local runs are unaffected.
    """
    if args.run_id is None and payload.get("run_id"):
        args.run_id = str(payload["run_id"])
    if args.dataset_name is None and payload.get("dataset_name"):
        args.dataset_name = str(payload["dataset_name"])
    if args.started_by_user_id is None and payload.get("started_by_user_id"):
        args.started_by_user_id = str(payload["started_by_user_id"])
    if args.started_by_display_name is None and payload.get("started_by_display_name"):
        args.started_by_display_name = str(payload["started_by_display_name"])
    if not args.models and payload.get("models"):
        args.models = [str(model) for model in payload["models"]]
    if args.fallback_model is None and payload.get("fallback_model"):
        args.fallback_model = str(payload["fallback_model"])
    # `is None`, not falsy: a user who deselected every derived feature
    # sends an explicit empty list, which must survive — not be treated as
    # "the config file didn't say" and fall back to the default selection.
    if args.derived_features is None and payload.get("derived_features") is not None:
        args.derived_features = [str(f) for f in payload["derived_features"]]
    if args.horizon is None and payload.get("horizon") is not None:
        horizon = int(payload["horizon"])
        if not (MIN_FORECAST_HORIZON <= horizon <= MAX_FORECAST_HORIZON):
            raise ConfigurationError(
                f"horizon must be between {MIN_FORECAST_HORIZON} and {MAX_FORECAST_HORIZON} (got {horizon})."
            )
        args.horizon = horizon


# Build a ForecastConfiguration from CLI args or config file
def build_configuration_from_args(args: argparse.Namespace) -> ForecastConfiguration:
    # Supports both entry paths the platform needs: a JSON file (the exact
    # payload the backend's Metadata Interpreter emits) or explicit flags
    # for local runs.
    if args.config:
        return ForecastConfiguration.from_dict(load_config_payload(args))

    configuration = ForecastConfiguration(
        date_column=args.date_column,
        target_column=args.target_column,
        key_columns=tuple(args.key_columns or ()),
        feature_columns=tuple(args.feature_columns or ()),
        aggregation_method=AggregationMethod(args.aggregation_method),
    )
    configuration.validate()
    return configuration


# Parse CLI arguments for a local run
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="forecast_engine.run_pipeline",
        description="Prepare any time-series dataset into forecast-ready series.",
    )
    parser.add_argument("--dataset", required=True, help="Path to the CSV/Excel dataset.")
    parser.add_argument("--config", help="Path to a JSON forecast configuration (from the backend).")
    parser.add_argument("--date-column", help="Column holding each observation's timestamp.")
    parser.add_argument("--target-column", help="Numeric column to forecast.")
    parser.add_argument("--key-columns", nargs="*", default=[], help="Business key column(s).")
    parser.add_argument("--feature-columns", nargs="*", default=[], help="Optional regressor column(s).")
    parser.add_argument(
        "--aggregation-method",
        choices=[method.value for method in AggregationMethod],
        default=AggregationMethod.SUM.value,
        help="How to roll a sub-monthly target up to monthly (default: sum).",
    )
    parser.add_argument("--run-id", help="Optional run identifier.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Display name for the dataset, recorded with the run (defaults to the file's own name).",
    )
    parser.add_argument(
        "--started-by-user-id",
        default=None,
        help="Stable identity of the user who submitted this run, recorded with the MLflow Parent Run.",
    )
    parser.add_argument(
        "--started-by-display-name",
        default=None,
        help="Display name for the user who submitted this run.",
    )
    parser.add_argument("--summary-out", help="Write the run summary JSON to this path.")
    parser.add_argument(
        "--live-status-out",
        default=None,
        help="Write the current stage trail to this path after every stage transition, for live progress polling.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Models to train (default: every registered model). e.g. --models arima xgboost",
    )
    parser.add_argument(
        "--fallback-model",
        default=None,
        help="Model used when every evaluated model fails validation (default: the configured baseline).",
    )
    parser.add_argument(
        "--derived-features",
        nargs="*",
        default=None,
        help=(
            "Derived feature columns (lag_*, rolling_mean_*, month, quarter) to generate for "
            "XGBoost/LightGBM (default: every supported feature, matching pre-existing behavior). "
            "Pass with no values to select none."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help=(
            f"Forward forecast horizon in months, {MIN_FORECAST_HORIZON}-{MAX_FORECAST_HORIZON} "
            "(default: the configured EvaluationConfig default, 12)."
        ),
    )
    parser.add_argument(
        "--azure-openai-endpoint",
        default=None,
        help=(
            "Sets AZURE_OPENAI_ENDPOINT for this process before the pipeline runs. Exists for "
            "compute (a Databricks Jobs API python_wheel_task) that cannot inject environment "
            "variables directly — a deployment that already sets the environment variable itself "
            "does not need this flag. Never logged."
        ),
    )
    parser.add_argument(
        "--azure-openai-api-key",
        default=None,
        help="Sets AZURE_OPENAI_API_KEY for this process before the pipeline runs. See --azure-openai-endpoint. Never logged.",
    )
    parser.add_argument(
        "--azure-openai-deployment",
        default=None,
        help="Sets AZURE_OPENAI_DEPLOYMENT_NAME for this process before the pipeline runs. See --azure-openai-endpoint. Never logged.",
    )
    parser.add_argument(
        "--parallel-keys",
        action="store_true",
        help=(
            "Run each forecast key's complete train/evaluate/explain/rank/select workflow as an "
            "independent Ray task, scheduled against the CPUs Ray actually finds. Requires Ray; "
            "omit to run every key sequentially in one process."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=PHASE_ORDER,
        default=None,
        help=(
            "Run only this phase as one Databricks task, resuming from --run-id's checkpoint "
            "(the first phase starts fresh instead). Omit to run the full pipeline in one process."
        ),
    )

    args = parser.parse_args(argv)

    # Either a config file or the two required column flags must be given.
    if not args.config and not (args.date_column and args.target_column):
        parser.error("Provide --config, or both --date-column and --target-column.")

    if args.horizon is not None and not (MIN_FORECAST_HORIZON <= args.horizon <= MAX_FORECAST_HORIZON):
        parser.error(f"--horizon must be between {MIN_FORECAST_HORIZON} and {MAX_FORECAST_HORIZON} (got {args.horizon}).")

    return args


# CLI entry point; returns a process exit code (0 success, 1 stage failure)
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # A no-op unless a --azure-openai-* flag is set — see
    # forecast_engine/core/databricks_secrets.py for who sets one.
    apply_azure_openai_cli_overrides(args)

    # Bound before the try so the post-run volume sync below can read it
    # even on the paths that leave the try early.
    config_payload: dict[str, Any] = {}

    try:
        configuration = build_configuration_from_args(args)
        # Run-level options may also travel inside --config; see
        # apply_config_run_options for why cloud execution needs that.
        config_payload = load_config_payload(args)
        apply_config_run_options(args, config_payload)

        # The same payload also carries pipeline-level blocks. Only
        # `curated_storage` is set by a caller today, and only for cloud
        # execution: its default root is *relative*, which on a Databricks
        # driver resolves against a working directory that is destroyed when
        # the job ends, so the curated dataset would not outlive the run.
        # The caller passes an already-resolved absolute path, which keeps
        # every storage decision outside the engine.
        pipeline_config = PipelineConfig.from_dict(config_payload) if config_payload else None

        evaluation_config = EvaluationConfig.default()
        if args.horizon is not None:
            # EvaluationConfig is frozen; forecast_horizon is the one value
            # this run overrides, so every stage that reads it — forward
            # forecast generation (EvaluationPipeline) and Final Production
            # Model Selection (ProductionSelectionPipeline) — sees the same
            # user-chosen horizon without needing it threaded separately.
            evaluation_config = replace(evaluation_config, forecast_horizon=args.horizon)

        # Resolved once, before any collaborator is built from it — see
        # apply_to_model_config()'s own docstring for why that ordering
        # matters. `args.derived_features is None` (never mentioned by
        # this run) returns ModelConfig.default() completely unchanged.
        model_config = apply_to_model_config(ModelConfig.default(), args.derived_features)

        pipeline = ForecastEnginePipeline(
            evaluation_config=evaluation_config,
            pipeline_config=pipeline_config,
            model_config=model_config,
            parallel_keys=args.parallel_keys,
        )
        if args.stage:
            # One Databricks task's phase — resumes from --run-id's
            # checkpoint under the same artifacts root every other run
            # output already lives under, never a hardcoded path.
            artifacts_root = (pipeline_config or PipelineConfig.default()).artifacts_mirror.root_dir
            context = pipeline.run_checkpointed_stage(
                args.stage,
                run_id=args.run_id,
                dataset_path=args.dataset,
                configuration=configuration,
                artifacts_root=artifacts_root,
                selected_models=args.models,
                fallback_model=args.fallback_model,
                dataset_name=args.dataset_name,
                live_status_path=args.live_status_out,
                started_by_user_id=args.started_by_user_id,
                started_by_display_name=args.started_by_display_name,
                derived_features=args.derived_features,
            )
        else:
            context = pipeline.run(
                args.dataset,
                configuration,
                run_id=args.run_id,
                selected_models=args.models,
                fallback_model=args.fallback_model,
                dataset_name=args.dataset_name,
                live_status_path=args.live_status_out,
                started_by_user_id=args.started_by_user_id,
                started_by_display_name=args.started_by_display_name,
                derived_features=args.derived_features,
            )
    except ForecastEngineError as exc:
        print(f"Forecast Engine failed: {exc}", file=sys.stderr)
        return 1

    summary = context.summary()
    print(json.dumps(summary, indent=2))

    if args.summary_out:
        storage.write_text(args.summary_out, json.dumps(summary, indent=2))

    # Last, so it carries summary.json across too. A no-op unless this run
    # executes inside a container image, which is the only case that cannot
    # write its outputs to their final home directly — see
    # forecast_engine/s03_storage/volume_sync.py.
    outcome = sync_outputs_to_volume(config_payload)
    if outcome is not None:
        print(f"Volume sync: {outcome.describe()}", file=sys.stderr)
        if not outcome.ok:
            # Reported as a failed run rather than swallowed. The forecast
            # itself succeeded and is in MLflow, but its results are not in
            # the storage account the platform treats as their home, and a
            # run that quietly half-lands there is exactly the failure mode
            # this whole step exists to end.
            print(
                "Forecast Engine failed: the run completed but its outputs could not be "
                "copied to the storage volume.",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
