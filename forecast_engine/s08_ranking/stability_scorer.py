"""Judges the shape of each forward forecast: variance, smoothness, intervals.

Independent of backtest accuracy — a model can score well on history and
still hand production an erratic forecast, and this is what catches that.

Measurements are normalized across one group's survivors, so every group is
scored on its own terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecast_engine.config.ranking_config import StabilityWeightsConfig
from forecast_engine.s06_evaluation.evaluation_report import ForwardForecast
from forecast_engine.s08_ranking.ranking_report import StabilityScoreBreakdown


@dataclass
class _RawStability:
    variance: float
    smoothness: float
    interval_width: float | None


class ForecastStabilityScorer:
    """Scores forecast stability for every candidate in one group."""

    def __init__(self, config: StabilityWeightsConfig | None = None) -> None:
        self._config = config or StabilityWeightsConfig()

    # Score every candidate's forecast, normalized within this group
    def score_group(self, forecasts: dict[str, ForwardForecast]) -> dict[str, StabilityScoreBreakdown]:
        raw = {name: self._measure(forecast) for name, forecast in forecasts.items()}

        # Lower variance/smoothness (less erratic) is better; higher score.
        variance_scores = _inverse_normalize({name: r.variance for name, r in raw.items()})
        smoothness_scores = _inverse_normalize({name: r.smoothness for name, r in raw.items()})

        # Narrower relative intervals indicate more confident predictions;
        # a model with no intervals gets the configured neutral score rather
        # than being penalized for a capability Phase 6 did not add.
        with_intervals = {name: r.interval_width for name, r in raw.items() if r.interval_width is not None}
        interval_scores_measured = _inverse_normalize(with_intervals)
        interval_scores = {
            name: interval_scores_measured.get(name, self._config.no_interval_score) for name in raw
        }

        results: dict[str, StabilityScoreBreakdown] = {}
        for name in raw:
            variance_score = variance_scores[name]
            smoothness_score = smoothness_scores[name]
            interval_score = interval_scores[name]

            score = (
                variance_score * self._config.variance_weight
                + smoothness_score * self._config.smoothness_weight
                + interval_score * self._config.interval_weight
            )
            weight_total = (
                self._config.variance_weight + self._config.smoothness_weight + self._config.interval_weight
            )
            score = score / weight_total if weight_total else 0.0

            results[name] = StabilityScoreBreakdown(
                variance_score=variance_score,
                smoothness_score=smoothness_score,
                interval_score=interval_score,
                score=score,
            )
        return results

    # Measure raw variance, smoothness and interval width for one forecast
    def _measure(self, forecast: ForwardForecast) -> _RawStability:
        values = np.asarray(forecast.values, dtype=float)
        values = values[np.isfinite(values)]

        if values.size == 0:
            return _RawStability(variance=0.0, smoothness=0.0, interval_width=None)

        # Coefficient of variation: dispersion relative to level, so the
        # measure is comparable across forecasts of different magnitude.
        mean = float(np.mean(values))
        variance = float(np.std(values) / abs(mean)) if mean != 0 else float(np.std(values))

        # Mean absolute second difference — how jagged the forecast path is,
        # independent of its overall level or trend.
        if values.size >= 3:
            second_diff = np.diff(values, n=2)
            smoothness = float(np.mean(np.abs(second_diff)) / abs(mean)) if mean != 0 else float(
                np.mean(np.abs(second_diff))
            )
        else:
            smoothness = 0.0

        interval_width = None
        if forecast.has_intervals:
            lower = np.asarray(forecast.lower, dtype=float)
            upper = np.asarray(forecast.upper, dtype=float)
            widths = upper - lower
            interval_width = float(np.mean(widths) / abs(mean)) if mean != 0 else float(np.mean(widths))

        return _RawStability(variance=variance, smoothness=smoothness, interval_width=interval_width)


# Min-max normalize, then invert so a lower raw value scores higher
def _inverse_normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}

    low = min(values.values())
    high = max(values.values())

    # A single candidate (or all-equal values) has nothing to compare
    # against, so it gets the max score rather than an undefined ratio.
    if high == low:
        return {name: 1.0 for name in values}

    return {name: 1.0 - ((value - low) / (high - low)) for name, value in values.items()}
