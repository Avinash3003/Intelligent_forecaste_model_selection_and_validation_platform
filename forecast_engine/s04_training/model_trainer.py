"""Model Trainer — trains every selected model on every forecasting group.

The orchestration layer of the training phase. For each group it walks the
user's selected models, tunes, fits, and records the outcome, producing one
TrainedModel per (group, model) pair.

Two behaviours matter most here:

  * Failure isolation. A model that fails on one group must not end the
    run — Section 5.3 requires failed keys to be flagged without blocking
    the rest. Every fit is therefore individually guarded, and the failure
    is recorded as a result rather than raised.
  * Selection fidelity. Only models the user chose are trained
    (Section 5.2), and availability is resolved once per run rather than
    per group, since it cannot change mid-run.

No evaluation happens here. Nothing is scored, compared, ranked or
forecast; that is the next phase's work. The output is trained objects and
training metadata, nothing more.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from forecast_engine.config.model_config import ModelConfig, ModelSpec
from forecast_engine.s05_models.base_model import TrainedModel, TrainingStatus
from forecast_engine.s05_models.model_registry import ModelRegistry
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.s04_training.hyperparameter_tuner import HyperparameterTuner


@dataclass
class TrainingReport:
    """Aggregate outcome of the training phase.

    Holds every TrainedModel plus counts by status, so a caller can judge a
    run at a glance without walking thousands of records.
    """

    results: list[TrainedModel] = field(default_factory=list)
    models_requested: list[str] = field(default_factory=list)
    models_unavailable: list[str] = field(default_factory=list)
    groups_trained: int = 0
    duration_seconds: float = 0.0

    # Count results that finished as successfully trained
    @property
    def trained_count(self) -> int:
        return sum(1 for result in self.results if result.status is TrainingStatus.TRAINED)

    # Count results that finished as failed
    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.results if result.status is TrainingStatus.FAILED)

    # Count results that were skipped
    @property
    def skipped_count(self) -> int:
        return sum(1 for result in self.results if result.status is TrainingStatus.SKIPPED)

    # Count results whose backing library was unavailable
    @property
    def unavailable_count(self) -> int:
        return sum(1 for result in self.results if result.status is TrainingStatus.UNAVAILABLE)

    # Only the successfully fitted models -- the next phase's input
    def trained_models(self) -> list[TrainedModel]:
        return [result for result in self.results if result.status is TrainingStatus.TRAINED]

    # Serializable summary; individual records are included in full
    def to_dict(self) -> dict[str, Any]:
        return {
            "models_requested": list(self.models_requested),
            "models_unavailable": list(self.models_unavailable),
            "groups_trained": self.groups_trained,
            "total_results": len(self.results),
            "trained": self.trained_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "unavailable": self.unavailable_count,
            "duration_seconds": round(self.duration_seconds, 3),
            "results": [result.to_dict() for result in self.results],
        }


class ModelTrainer:
    """Trains selected models across all forecasting groups."""

    # Wire up config, registry and tuner, defaulting each if not supplied
    def __init__(
        self,
        model_config: ModelConfig | None = None,
        registry: ModelRegistry | None = None,
        tuner: HyperparameterTuner | None = None,
    ) -> None:
        self._config = model_config or ModelConfig.default()
        self._registry = registry or ModelRegistry(self._config)
        self._tuner = tuner or HyperparameterTuner(self._config.tuning)

    # Train every selected model on every series, returning one report
    def train_all(
        self, series_collection: list[ForecastSeries], selected_models: list[str] | None = None
    ) -> TrainingReport:
        started = time.perf_counter()

        specs = self._registry.resolve_specs(selected_models)
        report = TrainingReport(models_requested=[spec.name for spec in specs])

        # Availability cannot change during a run, so it is resolved once
        # rather than re-checked for every one of potentially thousands of
        # groups.
        availability = {spec.name: self._registry.is_available(spec) for spec in specs}
        report.models_unavailable = [name for name, ok in availability.items() if not ok]

        for series in series_collection:
            for spec in specs:
                report.results.append(
                    self._train_one(spec, series, available=availability[spec.name])
                )
            report.groups_trained += 1

        report.duration_seconds = time.perf_counter() - started
        return report

    # Train one model on one group, capturing any failure as a result
    def _train_one(
        self, spec: ModelSpec, series: ForecastSeries, available: bool
    ) -> TrainedModel:
        record = TrainedModel(
            group_id=series.group_id,
            model_name=spec.name,
            status=TrainingStatus.SKIPPED,
            key_values=series.key_values,
        )

        # An uninstalled library is an expected deployment choice, not an
        # error — distinguish it so later stages can tell the two apart.
        if not available:
            record.status = TrainingStatus.UNAVAILABLE
            record.error = f"The library backing '{spec.name}' is not installed in this environment."
            return record

        # Too little history is a property of the key, not a defect; the
        # model is skipped for this group and trained for others.
        if series.observation_count < spec.min_observations:
            record.status = TrainingStatus.SKIPPED
            record.error = (
                f"Group has {series.observation_count} observation(s); "
                f"'{spec.name}' requires at least {spec.min_observations}."
            )
            return record

        started = time.perf_counter()
        try:
            model = self._registry.create(spec)
            model.initialize()

            tuned_params, tuning_metadata = self._tuner.tune(model, series)

            # Rebuild with the chosen parameters so the fitted estimator is
            # unambiguously the tuned one.
            model = self._registry.create(spec, tuned_params)
            model.initialize()
            training_metadata = model.train(series)

            record.status = TrainingStatus.TRAINED
            record.model = model.model
            record.fitted_model = model
            record.params = model.params
            record.metadata = {
                **training_metadata,
                "tuning": tuning_metadata,
                "duration_seconds": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:  # noqa: BLE001 - isolated per model by design
            record.status = TrainingStatus.FAILED
            record.error = f"{type(exc).__name__}: {exc}"
            record.metadata = {"duration_seconds": round(time.perf_counter() - started, 3)}

        return record
