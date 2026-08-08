"""Drift result models — the output contract of Sections 6.7–6.9.

Mirrors the pattern already established for evaluation and ranking: plain
dataclasses with a `to_dict()`, no narrative text, and the reason behind
every automated decision recorded as a short, structured string rather than
implied by a bare number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forecast_engine.config.drift_config import DriftAlgorithm, ThresholdMethod


@dataclass
class DriftAlgorithmSelection:
    """Outcome of Dynamic Drift Algorithm Selection (Section 6.7)."""

    algorithm: DriftAlgorithm
    reason: str
    sample_size: int
    unique_values: int
    normality_test: str | None = None
    normality_p_value: float | None = None
    is_normal: bool | None = None

    # Serialize to a plain dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm.value,
            "reason": self.reason,
            "sample_size": self.sample_size,
            "unique_values": self.unique_values,
            "normality_test": self.normality_test,
            "normality_p_value": _round(self.normality_p_value),
            "is_normal": self.is_normal,
        }


@dataclass
class ThresholdEstimate:
    """Outcome of Dynamic Threshold Estimation (Section 6.8)."""

    method: ThresholdMethod
    value: float
    reason: str
    null_sample_size: int = 0

    # Serialize to a plain dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "value": round(self.value, 6),
            "reason": self.reason,
            "null_sample_size": self.null_sample_size,
        }


@dataclass
class DriftValidationResult:
    """Outcome of validating one candidate's forecast against Section 6.9's
    Drift Validation step.

    `stage_trail` records the ordered sub-stages that produced this verdict
    — Dynamic Drift Selection, Threshold Validation, Drift Validation — so
    the execution order is auditable from the result itself rather than
    only readable in the code.
    """

    algorithm_selection: DriftAlgorithmSelection
    threshold: ThresholdEstimate
    drift_statistic: float
    passed: bool
    detail: str
    stage_trail: list[dict[str, Any]] = field(default_factory=list)

    # Serialize to a plain dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm_selection": self.algorithm_selection.to_dict(),
            "threshold": self.threshold.to_dict(),
            "drift_statistic": round(self.drift_statistic, 6),
            "passed": self.passed,
            "detail": self.detail,
            "stage_trail": self.stage_trail,
        }


# Round a nullable float for serialization
def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)
