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
from forecast_engine.core.live_status import LiveStatusWriter
from forecast_engine.core.pipeline_checkpoint import load_checkpoint, merge_checkpoints, save_checkpoint
from forecast_engine.core.pipeline_context import PipelineContext, StageStatus
from forecast_engine.core.pipeline_result import PipelineResultBuilder
from forecast_engine.s01_preprocessing.data_preprocessor import DataPreprocessor
from forecast_engine.s01_preprocessing.dataset_loader import DatasetLoader
from forecast_engine.s01_preprocessing.frequency_detector import FrequencyDetector
from forecast_engine.s01_preprocessing.group_generator import GroupGenerator
from forecast_engine.s01_preprocessing.series_builder import SeriesBuilder
from forecast_engine.s02_quality.quality_assessor import DataQualityAssessor
from forecast_engine.s03_storage.curated_writer import CuratedDatasetWriter
from forecast_engine.s03_storage.model_writer import WinningModelWriter
from forecast_engine.s03_storage.forecast_export_writer import ForecastExportWriter
from forecast_engine.s03_storage.artifacts_mirror_writer import ArtifactsMirrorWriter
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


class ForecastEnginePipeline:
    """Runs the engine end to end.

    Stages are injected rather than built inline, so a deployment or test can
    substitute one without touching the orchestration.
    """

    # Every private `_stage(context)` method below, grouped into the units
    # the multi-task Databricks Serverless workflow runs as separate tasks
    # (see `databricks/resources/forecast_job_serverless.yml`). Nothing
    # about a stage's own logic changes for `run_stage()` versus `run()` —
    # this dict only says which stages share a task.
    STAGE_GROUPS: dict[str, tuple[str, ...]] = {
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
        "persist_models": ("_persist_winning_models",),
        "export_forecasts": ("_export_forecasts",),
        "business_insights": ("_generate_business_insights",),
        "mirror_artifacts": ("_mirror_artifacts",),
        "mlflow_tracking": ("_track_to_mlflow",),
    }

    # Which stage group(s) a task loads its checkpoint from. Empty means
    # "start fresh" (the first task); more than one entry is the DAG's one
    # fork/join — Persist Winning Models and Export Forecasts both descend
    # from Rank & Select, and Business Insights needs both of their
    # contributions merged (see `pipeline_checkpoint.merge_checkpoints`).
    STAGE_GROUP_PREDECESSORS: dict[str, tuple[str, ...]] = {
        "load_prepare": (),
        "build_series": ("load_prepare",),
        "train_models": ("build_series",),
        "evaluate_models": ("train_models",),
        "explain_models": ("evaluate_models",),
        "rank_select": ("explain_models",),
        "persist_models": ("rank_select",),
        "export_forecasts": ("rank_select",),
        "business_insights": ("persist_models", "export_forecasts"),
        "mirror_artifacts": ("business_insights",),
        "mlflow_tracking": ("mirror_artifacts",),
    }

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
    ) -> None:
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

        try:
            self._load_dataset(context)
            self._detect_frequency(context)
            self._assess_quality(context)
            self._preprocess(context)
            self._persist_curated_dataset(context)
            self._verify_curated_dataset(context)
            self._generate_groups(context)
            self._build_series(context)
            self._train_models(context)
            self._evaluate_models(context)
            self._generate_explainability(context)
            self._select_production_models(context)
            self._persist_winning_models(context)
            self._export_forecasts(context)
            self._generate_business_insights(context)
            self._mirror_artifacts(context)
            self._track_to_mlflow(context)
        except Exception as exc:
            # The stage trail already records which stage failed; passing it
            # to MLflow as a tag makes a failed run searchable by where it
            # died, not just that it did.
            context.tracking_result = self._tracking_pipeline.fail(context.run_id, exc, _failed_stage(context))
            context.finish()
            raise

        context.finish()
        return context

    def run_stage(
        self,
        stage_group: str,
        *,
        checkpoint_dir: str | Path,
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
        """Run one stage group as an independent Databricks task.

        The multi-task workflow calls this once per task instead of run()'s
        single pass. Every stage method invoked is the same one run() calls,
        in the same order — this only adds checkpointing at task boundaries.

        load_prepare starts fresh and opens the MLflow run. Every later group
        loads its predecessor's checkpoint — two only for business_insights,
        the DAG's join after the parallel persist/export branches — and
        resumes the same MLflow run, so a failure anywhere still closes it
        with an accurate status.
        """
        if stage_group not in self.STAGE_GROUPS:
            raise ConfigurationError(
                f"Unknown stage group '{stage_group}'. Valid values: "
                f"{', '.join(self.STAGE_GROUPS)}."
            )

        if fallback_model:
            self._apply_fallback_model(fallback_model)

        checkpoint_dir = Path(checkpoint_dir)
        predecessors = self.STAGE_GROUP_PREDECESSORS[stage_group]

        if not predecessors:
            context = PipelineContext.create(
                dataset_path=dataset_path,
                configuration=configuration,
                pipeline_config=self._config,
                run_id=run_id,
            )
            context.selected_models = selected_models
            context.fallback_model = self._model_config.fallback_model
            context.derived_features = derived_features
            if live_status_path is not None:
                context.on_stage_change = LiveStatusWriter(live_status_path)
            context.tracking_result = self._tracking_pipeline.begin(
                context.run_id,
                dataset_name or Path(dataset_path).name,
                started_by_user_id=started_by_user_id,
                started_by_display_name=started_by_display_name,
            )
        else:
            contexts = [
                load_checkpoint(checkpoint_dir / f"after_{predecessor}.pkl")
                for predecessor in predecessors
            ]
            context = contexts[0]
            for other in contexts[1:]:
                merge_checkpoints(context, other)
            if live_status_path is not None:
                context.on_stage_change = LiveStatusWriter(live_status_path)
            resume_run_id = context.tracking_result.run_id if context.tracking_result else None
            self._tracking_pipeline.resume(resume_run_id)

        try:
            for method_name in self.STAGE_GROUPS[stage_group]:
                getattr(self, method_name)(context)
        except Exception as exc:
            context.tracking_result = self._tracking_pipeline.fail(context.run_id, exc, _failed_stage(context))
            context.finish()
            raise

        if stage_group == "mlflow_tracking":
            context.finish()
        else:
            save_checkpoint(context, checkpoint_dir / f"after_{stage_group}.pkl")

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
        record = context.begin_stage("Assess Data Quality")
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
        record = context.begin_stage("Persist Curated Dataset")
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
        record = context.begin_stage("Verify Curated Dataset")
        try:
            self._curated_validator.validate(context.prepared_dataset, context.configuration)
            context.complete_stage(record, "Curated dataset integrity verified.")
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Stage 7 — split the curated dataset into per-business-key groups
    def _generate_groups(self, context: PipelineContext) -> None:
        record = context.begin_stage("Generate Forecast Groups")
        try:
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
        record = context.begin_stage("Build Forecast Series")
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

    # Stage 9 — train every selected model on every forecasting group
    def _train_models(self, context: PipelineContext) -> None:
        record = context.begin_stage("Train Models")
        try:
            # Individual training failures are captured in the report rather
            # than raised, so one bad key or model never ends the run. Only
            # a configuration fault (unknown model, broken adapter) propagates.
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
            )
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Stage 10 — backtest, forecast forward and eliminate
    def _evaluate_models(self, context: PipelineContext) -> None:
        record = context.begin_stage("Evaluate Models")
        try:
            # Produces the surviving-model set that ranking consumes.
            # Individual (group, model) failures are captured in the report
            # rather than raised, so one bad pair never ends the run.
            trained = context.training_report.trained_models() if context.training_report else []
            report = self._evaluator.evaluate_all(context.series, trained)
            context.evaluation_report = report

            context.record(
                models_survived=report.survived_count,
                models_eliminated=report.eliminated_count,
                models_evaluation_failed=report.failed_count,
            )
            context.complete_stage(
                record,
                f"{report.survived_count:,} survived, {report.eliminated_count:,} eliminated, "
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

    # Stage 11 — SHAP / feature importance generation (Section 6.10)
    def _generate_explainability(self, context: PipelineContext) -> None:
        # Runs on every surviving (group, model) pair, strictly before Model
        # Ranking — its output is one of Ranking's composite inputs. Per-pair
        # failures are captured on the report by the engine itself, never
        # raised, so one model's explainability computation cannot block the
        # rest of its group.
        record = context.begin_stage("Generate Explainability (SHAP)")
        try:
            trained = context.training_report.trained_models() if context.training_report else []
            report = self._explainability_generator.generate_all(context.evaluation_report, trained, context.series)
            context.explainability_report = report

            context.record(explainability_results=len(report.results))
            context.complete_stage(
                record, f"Explainability generated for {len(report.results):,} surviving model(s)."
            )
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Stage 12 — rank survivors and select each group's production model
    def _select_production_models(self, context: PipelineContext) -> None:
        record = context.begin_stage("Rank & Select Production Models")
        try:
            # Ranking consumes Stage 11's ExplainabilityReport rather than
            # computing SHAP itself. Ranking and Final Production Model
            # Selection each isolate failure at their own grain (per group);
            # a configuration-level fault is the only thing that reaches
            # this try/except.
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
            context.complete_stage(
                record,
                f"{selection_report.selected_count:,} selected, {selection_report.fallback_count:,} used "
                f"the fallback model, {selection_report.unavailable_count:,} had no model available "
                f"across {len(selection_report.results):,} group(s).",
            )
        except ForecastEngineError as exc:
            context.fail_stage(record, exc)
            raise

    # Persist the winning fitted model for each forecast key
    def _persist_winning_models(self, context: PipelineContext) -> None:
        record = context.begin_stage("Persist Winning Models")
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
        record = context.begin_stage("Generate Business Insights")
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
        record = context.begin_stage("Track to MLflow")
        pipeline_result = PipelineResultBuilder().build(context)

        # The stage stays *open* across the tracking call, so anything
        # polling the live-status file sees "Track to MLflow — Running"
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
    payload = json.loads(Path(args.config).read_text())
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
            "compute (Databricks Serverless python_wheel_task) that cannot inject environment "
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
        "--stage-group",
        choices=list(ForecastEnginePipeline.STAGE_GROUPS),
        default=None,
        help=(
            "Run only this stage group instead of the whole pipeline, checkpointing "
            "PipelineContext via --checkpoint-dir for the next task to resume from. Exists for "
            "the multi-task Databricks Serverless workflow (see forecast_job_serverless.yml); "
            "omit to run the complete pipeline in one process, as local and DCS execution do."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for cross-task PipelineContext checkpoints. Required together with --stage-group.",
    )

    args = parser.parse_args(argv)

    # Either a config file or the two required column flags must be given.
    if not args.config and not (args.date_column and args.target_column):
        parser.error("Provide --config, or both --date-column and --target-column.")

    if args.horizon is not None and not (MIN_FORECAST_HORIZON <= args.horizon <= MAX_FORECAST_HORIZON):
        parser.error(f"--horizon must be between {MIN_FORECAST_HORIZON} and {MAX_FORECAST_HORIZON} (got {args.horizon}).")

    if args.stage_group is not None and not args.checkpoint_dir:
        parser.error("--stage-group requires --checkpoint-dir.")

    return args


# CLI entry point; returns a process exit code (0 success, 1 stage failure)
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Databricks-Serverless-only: resolves {{secrets/...}} job-parameter
    # values via dbutils.secrets.get() — see
    # forecast_engine/core/databricks_secrets.py. A no-op for local/DCS
    # execution, which already sets these as real environment variables.
    apply_azure_openai_cli_overrides(args)

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
            evaluation_config=evaluation_config, pipeline_config=pipeline_config, model_config=model_config
        )
        if args.stage_group is None:
            # The whole pipeline in one process — local execution and the
            # DCS job, unchanged from before the multi-task workflow existed.
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
        else:
            # One task of the multi-task Databricks Serverless workflow.
            context = pipeline.run_stage(
                args.stage_group,
                checkpoint_dir=args.checkpoint_dir,
                dataset_path=args.dataset,
                configuration=configuration,
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

    # An intermediate task's real output is the checkpoint run_stage()
    # already saved; the consolidated summary only exists once the last
    # task (mlflow_tracking) has run, exactly as it always has for a
    # single-process run.
    if args.stage_group is not None and args.stage_group != "mlflow_tracking":
        return 0

    summary = context.summary()
    print(json.dumps(summary, indent=2))

    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
