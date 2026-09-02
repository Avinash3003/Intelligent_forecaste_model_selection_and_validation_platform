"""Registers one model version per forecasting group.

Only the model final selection actually chose — ranked winner or fallback.
A rejected candidate is never registered.

Each business key has its own registered model, named
"{prefix}-{dataset}-{key}", stable across pipeline runs. A run publishes one
new version under that name, so a key's version history is its own selection
history rather than a stream shared with every other key of the dataset.

Registration is I/O-bound (~27s per key: artifact upload then a registry
round trip), so keys are registered with bounded parallelism, and a version
this run already created for a key is reused rather than duplicated.

The registered model wraps that group's already-computed forecast rather
than the raw estimator. The families trained here have incompatible native
prediction signatures, and a fallback's estimator is not retained past the
moment it forecast — so one wrapper flavour for every family avoids a
five-way branch and keeps this layer working from the result object alone.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

import pandas as pd

from forecast_engine.config.mlflow_config import MLflowConfig
from forecast_engine.core.pipeline_result import PipelineResult
from forecast_engine.s12_tracking.mlflow_client import MLflowClient, sanitize_model_name
from forecast_engine.utils.exceptions import MLflowTrackingError

import cloudpickle

from forecast_engine.s12_tracking.frozen_forecast_model import (
    PYFUNC_AVAILABLE as _PYFUNC_AVAILABLE,
    FrozenForecastModel,
)

# By value: the artifact must not depend on this wheel to load.
if _PYFUNC_AVAILABLE:
    from forecast_engine.s12_tracking import frozen_forecast_model as _wrapper_module

    cloudpickle.register_pickle_by_value(_wrapper_module)


def _signature_for(wrapper: "FrozenForecastModel") -> Any:
    """Unity Catalog refuses a model with no signature metadata."""
    from mlflow.models import infer_signature

    input_example = pd.DataFrame({"horizon": [float(len(wrapper.forecast.get("values", [])))]})
    return infer_signature(input_example, wrapper.predict(None))


@dataclass
class ModelRegistrationResult:
    """One group's Model Registry outcome — Section 6.13's required fields:
    Model Name, Model Version, Forecast Group, Run ID, Registration
    Timestamp.
    """

    group_id: str
    model_name: str
    run_id: str
    registered: bool
    registered_model_name: str | None = None
    model_version: str | None = None
    registered_at: str | None = None
    error: str | None = None
    # The registry metadata actually applied to this version, and any
    # failure to apply it. Separate from `error` on purpose: a model that
    # registered but could not be annotated is registered, and reporting
    # it as a registration failure would be a lie in the more alarming
    # direction.
    metadata_tags: dict[str, str] = field(default_factory=dict)
    tag_error: str | None = None

    # Convert to a plain dict for the tracking report
    def to_dict(self) -> dict[str, Any]:
        return {
            # JSON key stays "forecast_group": it is a frozen wire/tag name
            # already written into every past run's artifacts.
            "forecast_group": self.group_id,
            "model_name": self.model_name,
            "run_id": self.run_id,
            "registered": self.registered,
            "registered_model_name": self.registered_model_name,
            "model_version": self.model_version,
            "registered_at": self.registered_at,
            "error": self.error,
            "metadata_tags": self.metadata_tags,
            "tag_error": self.tag_error,
        }


# Register the Final Production Model for every forecasting group that has one
def register_winner_models(
    client: MLflowClient, pipeline_result: PipelineResult, config: MLflowConfig, run_id: str
) -> list[ModelRegistrationResult]:
    # One group's failure never blocks another's.
    if not config.register_winner_model:
        return []

    if not _PYFUNC_AVAILABLE:
        return [
            ModelRegistrationResult(
                group_id=winner["forecast_group"],
                model_name=winner.get("final_production_model") or "unknown",
                run_id=run_id,
                registered=False,
                error="The 'mlflow' package is not installed.",
            )
            for winner in pipeline_result.final_winner_models
        ]

    dataset_slug = _dataset_slug(pipeline_result, run_id)
    winners = pipeline_result.final_winner_models
    if not winners:
        return []

    # Bounded, and never more threads than keys.
    # One read for the whole run, not one lookup per key.
    published = client.published_versions()

    workers = max(1, min(config.registration_max_workers, len(winners)))
    if workers == 1:
        return [_register_one(client, w, run_id, config, dataset_slug, published) for w in winners]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # map preserves input order, so results stay deterministic.
        return list(pool.map(lambda w: _register_one(client, w, run_id, config, dataset_slug, published), winners))


# Stable dataset identity, the middle segment of the registered name.
def _dataset_slug(pipeline_result: PipelineResult, run_id: str) -> str:
    dataset_path = pipeline_result.dataset_metadata.get("dataset_path")
    name = PurePosixPath(dataset_path).stem if dataset_path else run_id
    return sanitize_model_name(name)


# Lineage the registry alone can answer selection questions from.
def _lineage_tags(group_id: str, model_name: str, run_id: str, winner: dict[str, Any]) -> dict[str, str]:
    return {
        "forecast_group": group_id,
        "model_name": model_name,
        "fallback_used": str(bool(winner.get("fallback_flag", False))),
        "selection_status": str(winner.get("final_selection_status") or "unknown"),
        "run_id": run_id,
    }


# Register a single group's winning model, returning its outcome
def _register_one(
    client: MLflowClient,
    winner: dict[str, Any],
    run_id: str,
    config: MLflowConfig,
    dataset_slug: str,
    published: dict[str, str],
) -> ModelRegistrationResult:
    group_id = winner["forecast_group"]
    model_name = winner.get("final_production_model")
    forecast = winner.get("forecast")

    if model_name is None or not forecast:
        return ModelRegistrationResult(
            group_id=group_id,
            model_name=model_name or "unknown",
            run_id=run_id,
            registered=False,
            error=(
                "No production model or forecast is available for this group "
                f"(Final Selection Status: {winner.get('final_selection_status')})."
            ),
        )

    group_slug = sanitize_model_name(group_id)
    registered_model_name = f"{config.registered_model_name_prefix}-{dataset_slug}-{group_slug}"

    tags = _lineage_tags(group_id, model_name, run_id, winner)
    artifact_path = f"model-{group_slug}"

    # Idempotent: this run already published this key, so do not republish.
    if group_slug in published:
        return ModelRegistrationResult(
            group_id=group_id,
            model_name=model_name,
            run_id=run_id,
            registered=True,
            registered_model_name=registered_model_name,
            model_version=published[group_slug] or None,
            registered_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            metadata_tags=tags,
        )

    wrapper = FrozenForecastModel(group_id=group_id, model_name=model_name, forecast=forecast)

    try:
        # Attached: worker threads have no active run of their own.
        with client.attached_run(run_id):
            # One path segment: artifact_path is part of model identity.
            model_info = client.register_pyfunc_model(
                python_model=wrapper,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                signature=_signature_for(wrapper),
            )
    except MLflowTrackingError as exc:
        return ModelRegistrationResult(
            group_id=group_id, model_name=model_name, run_id=run_id, registered=False, error=str(exc)
        )

    version = getattr(model_info, "registered_model_version", None)

    tag_error: str | None = None
    if version is None:
        tag_error = "No registered version was returned, so metadata could not be attached."
    else:
        try:
            client.set_model_version_tags(registered_model_name, str(version), tags)
        except MLflowTrackingError as exc:
            # Registered but not annotated is still registered.
            tag_error = str(exc)

    with client.attached_run(run_id):
        client.mark_published(group_slug, str(version) if version is not None else "")

    return ModelRegistrationResult(
        group_id=group_id,
        model_name=model_name,
        run_id=run_id,
        registered=True,
        registered_model_name=registered_model_name,
        model_version=str(version) if version is not None else None,
        registered_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        metadata_tags=tags if tag_error is None else {},
        tag_error=tag_error,
    )
