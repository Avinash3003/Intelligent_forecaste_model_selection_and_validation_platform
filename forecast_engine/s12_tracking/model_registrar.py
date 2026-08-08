"""Model Registry (Section 6.13, "Model Registry").

Registers exactly one model per forecasting group: whichever one Final
Production Model Selection (Section 6.9) actually chose, ranked winner or
configured fallback alike. Nothing else — a ranked-but-rejected candidate
is never registered.

The registered model is a `mlflow.pyfunc.PythonModel` wrapping that group's
already-computed forward forecast, rather than the raw fitted estimator.
This is a deliberate, uniform choice: the model families this platform
trains (statistical, tree-based, transformer, and the seasonal-naive
fallback) have incompatible native prediction signatures, and a fallback
model's own fitted estimator is not retained anywhere past the moment it
produced its forecast (Section 6.9). One wrapper flavour that packages the
frozen forecast the same way for every family avoids a five-way branch of
near-identical native-flavour registration code, and keeps this layer
working from `PipelineResult` alone — the "only log the contents of this
object" contract Section 6.13 requires, extended to registration as well
as logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from forecast_engine.config.mlflow_config import MLflowConfig
from forecast_engine.core.pipeline_result import PipelineResult
from forecast_engine.s12_tracking.mlflow_client import MLflowClient, sanitize_model_name
from forecast_engine.utils.exceptions import MLflowTrackingError

try:
    import mlflow.pyfunc

    _PYFUNC_AVAILABLE = True
except ImportError:
    _PYFUNC_AVAILABLE = False


if _PYFUNC_AVAILABLE:

    class _FrozenForecastModel(mlflow.pyfunc.PythonModel):
        """Serves one forecasting group's already-computed forward forecast.

        A record of "this is what the platform forecast for this group in
        this run", not a retrainable/re-servable live estimator — see the
        module docstring for why that scope is deliberate here.
        """

        # Store the frozen forecast for this group
        def __init__(self, forecast_group: str, model_name: str, forecast: dict[str, Any]) -> None:
            self.forecast_group = forecast_group
            self.model_name = model_name
            self.forecast = forecast

        # Deliberately untyped: MLflow's schema-inference machinery treats a
        # type-hinted `predict` as an opt-in to its own signature-validation
        # convention (`list[...]`-wrapped hints), which does not fit this
        # wrapper's single-instance-per-forecast-group shape. Omitting the
        # hint keeps behaviour identical to explicitly passing no schema.
        # Serve the frozen forecast values as a DataFrame
        def predict(self, context, model_input=None, params=None):
            values = self.forecast.get("values", [])
            return pd.DataFrame(
                {
                    "date": self.forecast.get("dates", []),
                    "value": values,
                    "lower": self.forecast.get("lower") or [None] * len(values),
                    "upper": self.forecast.get("upper") or [None] * len(values),
                }
            )


@dataclass
class ModelRegistrationResult:
    """One group's Model Registry outcome — Section 6.13's required fields:
    Model Name, Model Version, Forecast Group, Run ID, Registration
    Timestamp.
    """

    forecast_group: str
    model_name: str
    run_id: str
    registered: bool
    registered_model_name: str | None = None
    model_version: str | None = None
    registered_at: str | None = None
    error: str | None = None

    # Convert to a plain dict for the tracking report
    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_group": self.forecast_group,
            "model_name": self.model_name,
            "run_id": self.run_id,
            "registered": self.registered,
            "registered_model_name": self.registered_model_name,
            "model_version": self.model_version,
            "registered_at": self.registered_at,
            "error": self.error,
        }


# Register the Final Production Model for every forecasting group that has one
def register_winner_models(
    client: MLflowClient, pipeline_result: PipelineResult, config: MLflowConfig, run_id: str
) -> list[ModelRegistrationResult]:
    # Isolated per group: one group's registration failure is recorded on
    # its own result and never blocks another group's registration.
    if not config.register_winner_model:
        return []

    if not _PYFUNC_AVAILABLE:
        return [
            ModelRegistrationResult(
                forecast_group=winner["forecast_group"],
                model_name=winner.get("final_production_model") or "unknown",
                run_id=run_id,
                registered=False,
                error="The 'mlflow' package is not installed.",
            )
            for winner in pipeline_result.final_winner_models
        ]

    return [_register_one(client, winner, run_id, config) for winner in pipeline_result.final_winner_models]


# Register a single group's winning model, returning its outcome
def _register_one(
    client: MLflowClient, winner: dict[str, Any], run_id: str, config: MLflowConfig
) -> ModelRegistrationResult:
    group_id = winner["forecast_group"]
    model_name = winner.get("final_production_model")
    forecast = winner.get("forecast")

    if model_name is None or not forecast:
        return ModelRegistrationResult(
            forecast_group=group_id,
            model_name=model_name or "unknown",
            run_id=run_id,
            registered=False,
            error=(
                "No production model or forecast is available for this group "
                f"(Final Selection Status: {winner.get('final_selection_status')})."
            ),
        )

    group_slug = sanitize_model_name(group_id)
    registered_model_name = f"{config.registered_model_name_prefix}-{group_slug}"
    wrapper = _FrozenForecastModel(forecast_group=group_id, model_name=model_name, forecast=forecast)

    try:
        # `artifact_path` doubles as part of the model's identity when
        # `registered_model_name` is supplied, so it is held to the same
        # restricted charset as a registered model name (no "/", ".", etc.)
        # — a single path segment, not a nested artifact directory.
        model_info = client.register_pyfunc_model(
            python_model=wrapper,
            artifact_path=f"model-{group_slug}",
            registered_model_name=registered_model_name,
            tags={
                "forecast_group": group_id,
                "model_name": model_name,
                "fallback_used": str(bool(winner.get("fallback_flag", False))),
            },
        )
    except MLflowTrackingError as exc:
        return ModelRegistrationResult(
            forecast_group=group_id, model_name=model_name, run_id=run_id, registered=False, error=str(exc)
        )

    version = getattr(model_info, "registered_model_version", None)
    return ModelRegistrationResult(
        forecast_group=group_id,
        model_name=model_name,
        run_id=run_id,
        registered=True,
        registered_model_name=registered_model_name,
        model_version=str(version) if version is not None else None,
        registered_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
