"""What the tracking stage produces.

Kept on the context like every other stage's report, so a run's tracking
outcome (or its absence) is visible on the audit trail, even though nothing
here affects a forecasting decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forecast_engine.s12_tracking.model_registrar import ModelRegistrationResult


@dataclass
class TrackingResult:
    """Outcome of logging one pipeline execution to MLflow."""

    attempted: bool = False
    logged: bool = False
    status: str = "not_attempted"

    run_id: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    tracking_uri: str | None = None

    parameters_logged: int = 0
    metrics_logged: int = 0
    artifact_errors: dict[str, str] = field(default_factory=dict)
    registrations: list[ModelRegistrationResult] = field(default_factory=list)

    error: str | None = None
    duration_seconds: float = 0.0

    # Count how many groups got a registered model
    @property
    def models_registered(self) -> int:
        return sum(1 for registration in self.registrations if registration.registered)

    # Convert to a plain dict for the pipeline report
    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "logged": self.logged,
            "status": self.status,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "tracking_uri": self.tracking_uri,
            "parameters_logged": self.parameters_logged,
            "metrics_logged": self.metrics_logged,
            "artifact_errors": self.artifact_errors,
            "models_registered": self.models_registered,
            "registrations": [registration.to_dict() for registration in self.registrations],
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 3),
        }
