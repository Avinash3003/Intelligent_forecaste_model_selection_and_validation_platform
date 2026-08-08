"""Forecast confidence — an absolute quality measure, deliberately distinct
from the engine's ranking score.

The engine's `composite_ranking_score` (forecast_engine/s08_ranking) answers
"how did this model compare to the *other candidates in this run*" — it is
min-max normalized within each group's surviving cohort. With a single
survivor (a common case: a small dataset, or most candidates eliminated),
every component of that normalization trivially collapses to 1.0 regardless
of the model's actual accuracy, because there is nothing to compare it
against. Reusing that number as "confidence" therefore does not answer what
a user showed 100% actually wants to know: is this forecast any good.

Confidence here answers that question directly, from two things that are
true about a model on their own, independent of who else was in the race:

  * Backtest accuracy — 100% minus this model's own WMAPE from Section
    6.4's rolling backtest. An absolute error measure; a WMAPE of 8% means
    the same thing whether one model was evaluated or five.
  * Drift margin — how far below its dynamically estimated threshold
    (Section 6.8) the winning forecast's drift statistic landed. Passing
    by a hair and passing by a wide margin are both "Passed", but they are
    not equally reassuring, and this is the only place that distinction is
    surfaced.

A fallback model never undergoes drift validation (Section 6.9 — the
fallback path exists specifically because no candidate reached or survived
it), so its confidence is backtest accuracy alone, weights renormalized —
never a fabricated 0% (the old behaviour, since `composite_ranking_score`
is `None` on the fallback path) and never 100%.
"""

from __future__ import annotations

from dataclasses import dataclass

# Backtest accuracy is weighted higher: it is a direct, well-established
# accuracy measure, while drift margin is a validation gate's slack, a
# useful but secondary signal.
BACKTEST_WEIGHT = 0.7
DRIFT_WEIGHT = 0.3

FORMULA_BOTH = f"{BACKTEST_WEIGHT:.0%} × backtest accuracy + {DRIFT_WEIGHT:.0%} × drift margin"
FORMULA_BACKTEST_ONLY = "100% backtest accuracy (drift margin not applicable — fallback path)"


@dataclass(frozen=True)
class ConfidenceResult:
    """Confidence plus the evidence it was built from, so the UI can show
    the number and its justification together rather than a bare percentage.
    """

    confidence: float | None
    backtest_accuracy: float | None
    drift_margin: float | None
    formula: str
    explanation: str


def compute_confidence(
    *,
    wmape: float | None,
    drift_statistic: float | None,
    drift_threshold: float | None,
    is_fallback: bool,
) -> ConfidenceResult:
    """Compute an absolute confidence score from this model's own evidence.

    Args:
        wmape: The winning model's own backtest WMAPE, as a percentage
            (e.g. 8.2 for 8.2%) — matches `forecast_engine`'s own scale
            (s06_evaluation/metrics.py multiplies by 100).
        drift_statistic: The drift test statistic for the winning forecast,
            or None if drift validation never ran (fallback path).
        drift_threshold: The dynamically estimated threshold it was
            compared against.
        is_fallback: Whether this is the fallback path — used only to
            select the right formula/explanation text when both drift
            values are absent (as they always are for a fallback), so the
            "why" reads as "not applicable" rather than "missing data".

    Returns:
        A ConfidenceResult. `confidence` is None — never a fabricated
        number — when even backtest accuracy is unavailable.
    """
    backtest_accuracy = _backtest_accuracy(wmape)
    drift_margin = _drift_margin(drift_statistic, drift_threshold)

    if backtest_accuracy is None and drift_margin is None:
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
            formula="Not available",
            explanation=explanation,
        )

    if drift_margin is None:
        # Fallback path, or a winner whose drift record is otherwise
        # missing: weight is not silently dropped, it is renormalized onto
        # the one component that exists.
        confidence = backtest_accuracy
        formula = FORMULA_BACKTEST_ONLY if is_fallback else "100% backtest accuracy (drift data unavailable)"
        explanation = (
            f"Based only on backtest accuracy ({backtest_accuracy:.0%}) — this is the fallback "
            "model, which does not go through drift validation."
            if is_fallback
            else f"Based only on backtest accuracy ({backtest_accuracy:.0%}) — no drift record was found for this run."
        )
        return ConfidenceResult(confidence, backtest_accuracy, None, formula, explanation)

    if backtest_accuracy is None:
        # No plausible path today (a model reaching drift validation was
        # necessarily backtested first), kept only so this can never divide
        # by a smaller weight total than it reports.
        return ConfidenceResult(
            confidence=drift_margin,
            backtest_accuracy=None,
            drift_margin=drift_margin,
            formula="100% drift margin (backtest data unavailable)",
            explanation=f"Based only on drift margin ({drift_margin:.0%}) — no backtest metrics were recorded.",
        )

    confidence = BACKTEST_WEIGHT * backtest_accuracy + DRIFT_WEIGHT * drift_margin
    explanation = (
        f"{backtest_accuracy:.0%} backtest accuracy (from this model's own WMAPE of {wmape:.2f}%) "
        f"and a {drift_margin:.0%} drift margin (statistic {drift_statistic:.4g} against a threshold "
        f"of {drift_threshold:.4g}) combine to {confidence:.0%}."
    )
    return ConfidenceResult(confidence, backtest_accuracy, drift_margin, FORMULA_BOTH, explanation)


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
