"""Decides whether a forward forecast survives.

Historical accuracy alone does not determine the final model: one can score
well on backtest windows and still produce a forecast that is implausible
for the key it was fitted on. This stage catches exactly that.

Every rule runs even after one has failed, so the dashboard can show each
distinct reason a model was rejected rather than only the first.
"""

from __future__ import annotations

import numpy as np

from forecast_engine.config.evaluation_config import ForwardValidationConfig
from forecast_engine.s06_evaluation.evaluation_report import (
    ForwardForecast,
    ForwardValidationResult,
    RuleOutcome,
)
from forecast_engine.s06_evaluation.validation_rules import (
    AbnormalGrowthRule,
    ExcessiveSmoothingRule,
    ExcessiveVolatilityRule,
    FlatForecastRule,
    ForecastVarianceRule,
    IqrBoundsRule,
    MissingSeasonalityRule,
    MissingTrendRule,
    NonFiniteForecastRule,
    PercentileSpikeRule,
    ValidationRule,
)
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries


class ForwardForecastValidator:
    """Applies the elimination framework to a forward forecast."""

    # Wire up config and assemble the enabled rule set
    def __init__(self, config: ForwardValidationConfig | None = None) -> None:
        self._config = config or ForwardValidationConfig()
        self._rules = self._build_rules()

    # Judge a forward forecast against the history it was produced from
    def validate(self, series: ForecastSeries, forecast: ForwardForecast) -> ForwardValidationResult:
        result = ForwardValidationResult()

        if not self._config.enabled:
            result.skipped_reason = "Forward validation disabled by configuration."
            return result

        history = np.asarray(
            series.frame[series.target_column].to_numpy(), dtype=float
        )
        history = history[np.isfinite(history)]

        # Below this much history the rules would be comparing a forecast
        # against noise, so the model passes through to ranking unjudged.
        if len(history) < self._config.min_history_for_validation:
            result.skipped_reason = (
                f"Series has {len(history)} observation(s); at least "
                f"{self._config.min_history_for_validation} are required to validate a forecast."
            )
            return result

        values = np.asarray(forecast.values, dtype=float)
        if values.size == 0:
            result.skipped_reason = "Forecast contains no values to validate."
            return result

        # Every rule runs even after a failure, so the user sees all the
        # reasons a model was rejected rather than only the first.
        for rule in self._rules:
            try:
                result.rule_outcomes.append(rule.evaluate(history, values))
            except Exception as exc:  # noqa: BLE001 - a broken rule must not eliminate a model
                # A rule that cannot be computed abstains: eliminating a
                # model because our own check malfunctioned would be wrong.
                result.rule_outcomes.append(
                    RuleOutcome(
                        rule_id=rule.rule_id,
                        rule_name=rule.rule_name,
                        passed=True,
                        detail=f"Rule could not be evaluated: {type(exc).__name__}: {exc}",
                    )
                )

        result.passed = not result.failed_rules
        return result

    # Assemble the configured (enabled) rule set
    def _build_rules(self) -> list[ValidationRule]:
        overfitting = self._config.overfitting
        underfitting = self._config.underfitting

        rules: list[ValidationRule] = [
            NonFiniteForecastRule(),
            # Overfitting signals (Section 6.5.1)
            ExcessiveVolatilityRule(overfitting),
            PercentileSpikeRule(overfitting),
            IqrBoundsRule(overfitting),
            AbnormalGrowthRule(overfitting),
            ForecastVarianceRule(overfitting),
            # Underfitting signals (Section 6.5.2)
            FlatForecastRule(underfitting),
            ExcessiveSmoothingRule(underfitting),
            MissingSeasonalityRule(underfitting),
            MissingTrendRule(underfitting),
        ]
        return [rule for rule in rules if rule.is_enabled()]
