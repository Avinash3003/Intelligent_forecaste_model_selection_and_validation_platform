"""How good is this forecast, on its own terms.

Deliberately not the engine's composite ranking score: that one is
normalized within a run's surviving candidates, so a lone survivor scores
1.0 no matter how bad it is. Confidence instead blends three things that
are true of a model independently of who else was in the race:

  - backtest accuracy: 100% minus its own WMAPE.
  - forecast stability: does the forward forecast vary like the history?
    Catches the model that backtests well then emits a flat line.
  - drift margin: how far under its threshold the drift statistic landed.

A fallback never runs drift validation, so that weight is renormalized away
rather than counted as zero. Its stability is still measured, because the
fallback path is exactly where a flat forecast is most likely to ship.
"""

from __future__ import annotations

from dataclasses import dataclass

# Backtest accuracy stays the largest single component: it is a direct,
# well-established accuracy measure. Stability outweighs drift margin
# because it is a property of the forecast actually being shipped, where
# drift margin is a gate's slack.
#
# Any component that cannot be computed has its weight renormalized across
# the rest, never counted as zero — a missing measurement is not evidence
# of a bad forecast.
BACKTEST_WEIGHT = 0.5
STABILITY_WEIGHT = 0.3
DRIFT_WEIGHT = 0.2

_LABELS = {
    "backtest": "backtest accuracy",
    "stability": "forecast stability",
    "drift": "drift margin",
}
_WEIGHTS = {"backtest": BACKTEST_WEIGHT, "stability": STABILITY_WEIGHT, "drift": DRIFT_WEIGHT}

# How far a forecast's variation may sit from its history's before
# stability is scored at zero. The measure is symmetric in ratio space, so
# a forecast with 4% of history's variation and one with 25x it score
# alike — both are equally wrong about how much this series moves.
MAX_VARIATION_RATIO = 4.0


@dataclass(frozen=True)
class ConfidenceResult:
    """The score plus the evidence behind it, so the UI can justify the number."""

    confidence: float | None
    backtest_accuracy: float | None
    drift_margin: float | None
    formula: str
    explanation: str
    forecast_stability: float | None = None


def compute_confidence(
    *,
    wmape: float | None,
    drift_statistic: float | None,
    drift_threshold: float | None,
    is_fallback: bool,
    forecast_values: list[float] | None = None,
    history_values: list[float] | None = None,
) -> ConfidenceResult:
    """Score one model from its own evidence.

    wmape is a percentage (8.2 means 8.2%), matching the engine's scale.
    drift_statistic/drift_threshold are None on the fallback path, where
    drift validation never ran. history_values is the group's full observed
    history, which the forecast's variation is judged against.

    confidence is None — never a made-up number — if nothing was measurable.
    """
    scores: dict[str, float] = {}

    backtest_accuracy = _backtest_accuracy(wmape)
    if backtest_accuracy is not None:
        scores["backtest"] = backtest_accuracy

    stability = _forecast_stability(forecast_values, history_values)
    if stability is not None:
        scores["stability"] = stability

    drift_margin = _drift_margin(drift_statistic, drift_threshold)
    if drift_margin is not None:
        scores["drift"] = drift_margin

    if not scores:
        explanation = (
            "The fallback baseline is excluded from the normal candidate registry by design "
            "(Section 6.9) — it is trained on demand only when every ranked candidate fails, and is "
            "never backtested, so no accuracy evidence exists to compute a confidence score from."
            if is_fallback
            else "No backtest metrics were recorded for this model, so confidence cannot be computed."
        )
        return ConfidenceResult(
            confidence=None,
            backtest_accuracy=None,
            drift_margin=None,
            forecast_stability=None,
            formula="Not available",
            explanation=explanation,
        )

    # Renormalized across whichever components exist, so an absent
    # measurement dilutes nothing and is never scored as zero.
    total_weight = sum(_WEIGHTS[name] for name in scores)
    confidence = sum(_WEIGHTS[name] * value for name, value in scores.items()) / total_weight

    formula = " + ".join(
        f"{_WEIGHTS[name] / total_weight:.0%} x {_LABELS[name]}" for name in _WEIGHTS if name in scores
    )
    if "drift" not in scores:
        formula += " (drift margin not applicable - fallback path)" if is_fallback else " (drift data unavailable)"

    return ConfidenceResult(
        confidence=confidence,
        backtest_accuracy=backtest_accuracy,
        drift_margin=drift_margin,
        forecast_stability=stability,
        formula=formula,
        explanation=_explain(scores, confidence, wmape, drift_statistic, drift_threshold, is_fallback),
    )


def _explain(
    scores: dict[str, float],
    confidence: float,
    wmape: float | None,
    drift_statistic: float | None,
    drift_threshold: float | None,
    is_fallback: bool,
) -> str:
    """One sentence naming every component that contributed, with its
    underlying evidence — so a low score always says which part was weak."""
    parts: list[str] = []
    if "backtest" in scores:
        parts.append(f"{scores['backtest']:.0%} backtest accuracy (WMAPE {wmape:.2f}%)")
    if "stability" in scores:
        parts.append(
            f"{scores['stability']:.0%} forecast stability (how closely the forward forecast's "
            "variation matches this series' own history)"
        )
    if "drift" in scores:
        parts.append(
            f"a {scores['drift']:.0%} drift margin (statistic {drift_statistic:.4g} against a "
            f"threshold of {drift_threshold:.4g})"
        )

    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"
    sentence = f"{joined} combine to {confidence:.0%}."
    if is_fallback:
        sentence += (
            " This is the fallback baseline, which does not go through drift validation, so that "
            "component is excluded rather than counted against it."
        )
    return sentence


def _forecast_stability(
    forecast_values: list[float] | None, history_values: list[float] | None
) -> float | None:
    """Does the forecast vary like the history?

    Compares the two standard deviations symmetrically, so moving far less
    than history (a flat forecast) and far more (excessive volatility) are
    penalised alike. None when either side is too short to have a spread.
    """
    if not forecast_values or not history_values:
        return None
    if len(forecast_values) < 2 or len(history_values) < 2:
        return None

    history_spread = _stdev(history_values)
    forecast_spread = _stdev(forecast_values)
    if history_spread is None or forecast_spread is None:
        return None

    # A genuinely flat history (a constant series) makes the ratio
    # meaningless rather than infinite; a flat forecast is the correct
    # answer there, so stability is simply not scored.
    if history_spread <= 0:
        return None

    ratio = forecast_spread / history_spread
    if ratio <= 0:
        return 0.0

    # Symmetric: ratio and 1/ratio score identically. 1.0 at a perfect
    # match, decaying to 0 at MAX_VARIATION_RATIO in either direction.
    closeness = min(ratio, 1.0 / ratio)
    floor = 1.0 / MAX_VARIATION_RATIO
    if closeness <= floor:
        return 0.0
    return max(0.0, min(1.0, (closeness - floor) / (1.0 - floor)))


def _stdev(values: list[float]) -> float | None:
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    if len(numeric) < 2:
        return None
    mean = sum(numeric) / len(numeric)
    return (sum((v - mean) ** 2 for v in numeric) / len(numeric)) ** 0.5


def _backtest_accuracy(wmape: float | None) -> float | None:
    if wmape is None:
        return None
    # Clamped, not just scaled: a pathological model can have WMAPE well
    # over 100%, which must not render as "negative accuracy".
    return max(0.0, min(1.0, (100.0 - wmape) / 100.0))


def _drift_margin(statistic: float | None, threshold: float | None) -> float | None:
    if statistic is None or threshold is None or threshold <= 0:
        return None
    # Matches the engine's own pass rule exactly (drift_validator.py:131,
    # `passed = statistic <= threshold`): margin is 1.0 at statistic=0,
    # crosses 0.0 exactly at the pass/fail boundary, and is clamped rather
    # than going negative for a statistic beyond the threshold (which would
    # not be a production winner in the first place, but a rejected
    # candidate's margin is never computed through this path anyway).
    return max(0.0, min(1.0, 1.0 - statistic / threshold))
