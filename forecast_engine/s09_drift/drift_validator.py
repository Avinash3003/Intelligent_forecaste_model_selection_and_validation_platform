"""The gate a ranked candidate must clear to reach production.

Three separately reported sub-stages, in order:

    select algorithm  -> which statistic suits this group's data
    estimate threshold -> the cut-off for that statistic
    validate          -> statistic vs. threshold

Selection must precede the threshold: a threshold is a percentile of the
selected statistic's null distribution, so it only means anything once the
statistic is known. Each sub-stage appends to stage_trail.

Failing rejects that one candidate; it never terminates the run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from forecast_engine.config.drift_config import DriftValidationConfig
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.s06_evaluation.evaluation_report import ForwardForecast
from forecast_engine.s06_evaluation.reference_window import recent_reference_slice
from forecast_engine.s09_drift.algorithm_selector import DriftAlgorithmSelector
from forecast_engine.s09_drift.drift_algorithms import DRIFT_ALGORITHMS
from forecast_engine.s09_drift.drift_report import (
    DriftAlgorithmSelection,
    DriftValidationResult,
    ThresholdEstimate,
)
from forecast_engine.s09_drift.threshold_estimator import ThresholdEstimator


class DriftValidator:
    """Validates one candidate's forward forecast against its own history."""

    # Wire up config, the algorithm selector and the threshold estimator
    def __init__(
        self,
        config: DriftValidationConfig | None = None,
        selector: DriftAlgorithmSelector | None = None,
        threshold_estimator: ThresholdEstimator | None = None,
    ) -> None:
        self._config = config or DriftValidationConfig.default()
        self._selector = selector or DriftAlgorithmSelector(self._config.algorithm_selection)
        self._threshold_estimator = threshold_estimator or ThresholdEstimator(self._config.threshold_estimation)

    # Judge a forecast against the history it was produced from
    def validate(self, series: ForecastSeries, forecast: ForwardForecast) -> DriftValidationResult:
        # Raises ValueError if there is no usable historical or forecast
        # data to compare — the caller (Winner Selection) catches this per
        # candidate, never letting one candidate's unusable data stop
        # evaluation of the next.
        history, current = self._extract_distributions(series, forecast)

        trail: list[dict[str, Any]] = []
        selection = self._select_drift_algorithm(history, trail)
        threshold = self._validate_threshold(selection, history, trail)
        return self._validate_drift(selection, threshold, history, current, trail)

    # Stage inputs — the two distributions being compared.
    def _extract_distributions(
        self, series: ForecastSeries, forecast: ForwardForecast
    ) -> tuple[np.ndarray, np.ndarray]:
        full_history = np.asarray(series.frame[series.target_column].to_numpy(), dtype=float)
        full_history = full_history[np.isfinite(full_history)]
        if full_history.size == 0:
            raise ValueError("Series has no finite historical observations to validate drift against.")

        current = np.asarray(forecast.values, dtype=float)
        current = current[np.isfinite(current)]
        if current.size == 0:
            raise ValueError("Forecast has no finite values to validate.")

        # A recent, single-regime slice, not the entire series — the whole
        # history would judge a forecast against price/volume levels the
        # series may have long since moved past (reference_window.py). This
        # same `history` return value also feeds algorithm selection and
        # threshold estimation below, so the comparison population and the
        # threshold's calibration population are always drawn from the
        # identical window — deliberately, so the two stay statistically
        # consistent with each other.
        history = recent_reference_slice(full_history, series.frequency, current.size)
        return history, current

    # Stage 1 — Dynamic Drift Selection (Section 6.7).
    def _select_drift_algorithm(
        self, history: np.ndarray, trail: list[dict[str, Any]]
    ) -> DriftAlgorithmSelection:
        selection = self._selector.select(history)
        trail.append(
            {
                "stage": "Dynamic Drift Selection",
                "algorithm": selection.algorithm.value,
                "reason": selection.reason,
            }
        )
        return selection

    # Stage 2 — Threshold Validation (Section 6.8).
    def _validate_threshold(
        self,
        selection: DriftAlgorithmSelection,
        history: np.ndarray,
        trail: list[dict[str, Any]],
    ) -> ThresholdEstimate:
        threshold = self._threshold_estimator.estimate(selection.algorithm, history)
        # A zero threshold means the history was too thin to build any null
        # comparison; it is valid but maximally strict, so it is recorded
        # rather than silently treated as an ordinary cut-off.
        trail.append(
            {
                "stage": "Threshold Validation",
                "method": threshold.method.value,
                "value": round(threshold.value, 6),
                "reason": threshold.reason,
                "usable": threshold.value > 0.0,
            }
        )
        return threshold

    # Stage 3 — Drift Validation (Section 6.9).
    def _validate_drift(
        self,
        selection: DriftAlgorithmSelection,
        threshold: ThresholdEstimate,
        history: np.ndarray,
        current: np.ndarray,
        trail: list[dict[str, Any]],
    ) -> DriftValidationResult:
        statistic_fn = DRIFT_ALGORITHMS[selection.algorithm]
        statistic = float(statistic_fn(history, current))

        passed = statistic <= threshold.value
        detail = (
            f"{selection.algorithm.value} statistic {statistic:.6f} "
            f"{'<=' if passed else '>'} threshold {threshold.value:.6f} "
            f"({threshold.method.value})."
        )
        trail.append(
            {
                "stage": "Drift Validation",
                "statistic": round(statistic, 6),
                "threshold": round(threshold.value, 6),
                "passed": passed,
            }
        )

        return DriftValidationResult(
            algorithm_selection=selection,
            threshold=threshold,
            drift_statistic=statistic,
            passed=passed,
            detail=detail,
            stage_trail=trail,
        )
