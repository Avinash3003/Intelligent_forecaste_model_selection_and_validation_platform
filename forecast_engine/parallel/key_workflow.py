"""One forecast key's per-stage workflow, and the merge of many keys' output.

    train -> evaluate -> explain -> rank -> select

Each stage below is its own function, taking exactly the prior stage's real
output as input — never the whole key bundled into one call. This is what
lets `ray_executor.StagedKeyExecution` fan every key out per STAGE (all
keys train, then all keys evaluate, ...) instead of per KEY (one task doing
a key's whole workflow) — a genuine stage boundary, not a UI-only label over
one monolithic task.

No forecasting logic is reimplemented here — this only narrows the batch
each call covers from every key to exactly one, which is valid because
every one of these stages already scores a key against itself alone (see
`ModelRanker`'s module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forecast_engine.config.drift_config import DriftValidationConfig
from forecast_engine.config.evaluation_config import EvaluationConfig
from forecast_engine.config.explainability_config import ExplainabilityConfig
from forecast_engine.config.model_config import ModelConfig
from forecast_engine.config.ranking_config import RankingConfig
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.s04_training.model_trainer import ModelTrainer, TrainingReport
from forecast_engine.s05_models.base_model import TrainedModel
from forecast_engine.s05_models.model_registry import ModelRegistry
from forecast_engine.s06_evaluation.evaluation_pipeline import EvaluationPipeline
from forecast_engine.s06_evaluation.evaluation_report import EvaluationReport
from forecast_engine.s07_explainability.explainability_pipeline import ExplainabilityPipeline
from forecast_engine.s07_explainability.explainability_report import ExplainabilityReport
from forecast_engine.s08_ranking.ranking_report import RankingReport
from forecast_engine.s10_selection.production_pipeline import ProductionSelectionPipeline
from forecast_engine.s10_selection.selection_report import ProductionSelectionReport


@dataclass(frozen=True)
class KeyWorkflowConfig:
    """The immutable configuration every key's workflow shares.

    Put into Ray's object store once per run rather than serialized per
    task, since it is identical for every key.
    """

    model: ModelConfig
    evaluation: EvaluationConfig
    explainability: ExplainabilityConfig
    ranking: RankingConfig
    drift: DriftValidationConfig
    selected_models: tuple[str, ...] | None = None


@dataclass
class KeyReports:
    """The five stage reports for one key — or for many keys, merged.

    Deliberately the same type either way: a merge of one key's reports is
    that key's reports, so the stages that read this cannot tell whether the
    run fanned out or not.
    """

    training: TrainingReport = field(default_factory=TrainingReport)
    evaluation: EvaluationReport = field(default_factory=EvaluationReport)
    explainability: ExplainabilityReport = field(default_factory=ExplainabilityReport)
    ranking: RankingReport = field(default_factory=RankingReport)
    selection: ProductionSelectionReport = field(default_factory=ProductionSelectionReport)


# Stage 1 for a single key — no dependency on any other stage
def train_key(series: ForecastSeries, config: KeyWorkflowConfig) -> TrainingReport:
    selected = None if config.selected_models is None else list(config.selected_models)
    return ModelTrainer(config.model).train_all([series], selected)


# Stage 2 for a single key — depends only on that key's own training report
def evaluate_key(
    series: ForecastSeries, config: KeyWorkflowConfig, trained: list[TrainedModel]
) -> EvaluationReport:
    return EvaluationPipeline(ModelRegistry(config.model), config.evaluation).evaluate_all([series], trained)


# Stage 3 for a single key — depends on that key's evaluation and training
def explain_key(
    series: ForecastSeries,
    config: KeyWorkflowConfig,
    evaluation: EvaluationReport,
    trained: list[TrainedModel],
) -> ExplainabilityReport:
    return ExplainabilityPipeline(ModelRegistry(config.model), config.explainability).generate_all(
        evaluation, trained, [series]
    )


# Stage 4 for a single key — depends on that key's evaluation and explainability
def rank_select_key(
    series: ForecastSeries,
    config: KeyWorkflowConfig,
    evaluation: EvaluationReport,
    explainability: ExplainabilityReport,
) -> tuple[RankingReport, ProductionSelectionReport]:
    return ProductionSelectionPipeline(
        ModelRegistry(config.model),
        config.ranking,
        config.drift,
        config.model,
        forecast_horizon=config.evaluation.forecast_horizon,
    ).run(evaluation, explainability, [series])


# The sequential reference path: every stage for one key, one after another,
# in this process. Ray's staged path (ray_executor.StagedKeyExecution)
# calls the same four functions above, fanned out across keys per stage —
# same code, different scheduling, so the two are provably identical.
def run_key(series: ForecastSeries, config: KeyWorkflowConfig) -> KeyReports:
    training = train_key(series, config)
    trained = training.trained_models()

    evaluation = evaluate_key(series, config, trained)
    explainability = explain_key(series, config, evaluation, trained)
    ranking, selection = rank_select_key(series, config, evaluation, explainability)

    return KeyReports(training, evaluation, explainability, ranking, selection)


def merge_key_reports(per_key: list[KeyReports]) -> KeyReports:
    """Combine per-key reports into the run-level reports the stages record.

    Every field involved is additive — result lists append and `rankings` is
    keyed by group_id — so no key's output can displace another's, and every
    count is a derived property that follows from the merged lists. Caller
    order decides merged order; passing keys in series order reproduces
    exactly what a sequential run would have built.
    """
    merged = KeyReports()

    for reports in per_key:
        _merge_training(merged.training, reports.training)
        _merge_evaluation(merged.evaluation, reports.evaluation)
        merged.explainability.results.extend(reports.explainability.results)
        merged.explainability.duration_seconds += reports.explainability.duration_seconds
        merged.ranking.rankings.update(reports.ranking.rankings)
        merged.ranking.duration_seconds += reports.ranking.duration_seconds
        merged.selection.results.extend(reports.selection.results)
        merged.selection.duration_seconds += reports.selection.duration_seconds

    merged.evaluation.groups_evaluated = len({r.group_id for r in merged.evaluation.results})
    return merged


# Fold one key's training report into the run-level one
def _merge_training(into: TrainingReport, part: TrainingReport) -> None:
    into.results.extend(part.results)
    into.groups_trained += part.groups_trained
    into.duration_seconds += part.duration_seconds
    # Identical for every key (resolved from config, not from data), so the
    # first key to report them settles both lists.
    if not into.models_requested:
        into.models_requested = list(part.models_requested)
    if not into.models_unavailable:
        into.models_unavailable = list(part.models_unavailable)


# Fold one key's evaluation report, including its timing breakdown
def _merge_evaluation(into: EvaluationReport, part: EvaluationReport) -> None:
    into.results.extend(part.results)
    for attr in (
        "duration_seconds",
        "backtest_seconds",
        "forecast_generation_seconds",
        "validation_seconds",
        "model_fit_count",
        "backtest_windows_evaluated",
        "forecasts_reused",
        "forecasts_refit",
    ):
        setattr(into, attr, getattr(into, attr) + getattr(part, attr))
