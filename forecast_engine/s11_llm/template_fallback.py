"""Non-LLM explanation fallback (Section 11, "template-based, non-LLM
explanation path... defined for use if the primary endpoint is unavailable
or rate-limited").

Produces the exact same `InsightPayload` shape the LLM path produces, built
directly from the same structured facts the LLM would have been given —
deterministic, instant, and by construction perfectly grounded, since it
never writes a number that isn't a straight substitution from the metrics
dict.

This is what keeps Objective 1's hardest requirement true: the forecast
pipeline must finish with an explanation even when Azure OpenAI is fully
unavailable, rate-limited, or misconfigured. Nothing downstream can tell
the difference in shape between this and an LLM-generated payload — only
`provider` on the trace record says which one ran.
"""

from __future__ import annotations

from forecast_engine.s11_llm.schema import InsightPayload


def build_template_insight(
    *,
    selected_model: str,
    wmape: float | None,
    is_fallback: bool,
    fallback_trigger: str | None,
    rejected_candidates: list[dict[str, str]],
    confidence_pct: float | None,
    caveats: list[str],
    max_rejection_reasons: int = 6,
    max_caveats: int = 3,
) -> InsightPayload:
    """Build a structured insight with no LLM call.

    Args:
        selected_model: The final production model's name.
        wmape: The winning model's own backtest WMAPE (percentage scale),
            or None if unavailable (e.g. the fallback path never backtests).
        is_fallback: Whether the configured fallback model was used.
        fallback_trigger: The one-line reason every candidate failed, when
            `is_fallback` is True.
        rejected_candidates: `[{"model_name": ..., "reason": ...}, ...]`
            for every candidate that did not win, in the ranked order they
            were tried.
        confidence_pct: The platform's own computed confidence, 0-100.
        caveats: Pre-computed caveat phrases (short history, high
            missingness, fallback used) — this function does not derive
            them, it only carries them through.
    """
    summary = _summary_sentence(selected_model, wmape, is_fallback, fallback_trigger)

    reasons = [
        f"{item.get('model_name', 'a candidate')}: {item.get('reason', 'did not pass validation')}"
        for item in rejected_candidates[:max_rejection_reasons]
    ]

    return InsightPayload(
        selected_model=selected_model,
        rejection_reasons=reasons,
        confidence=round(confidence_pct, 1) if confidence_pct is not None else None,
        caveats=list(caveats[:max_caveats]),
        concise_summary=summary,
    )


def _summary_sentence(
    selected_model: str, wmape: float | None, is_fallback: bool, fallback_trigger: str | None
) -> str:
    if is_fallback:
        reason = fallback_trigger or "every evaluated candidate failed validation"
        return f"{selected_model} was used as the fallback baseline because {reason.rstrip('.')}."

    if wmape is not None:
        return (
            f"{selected_model} was selected, with a backtest WMAPE of {wmape:.2f}%, meaning its "
            f"backtested forecasts were off by about {wmape:.0f}% on average."
        )

    return f"{selected_model} was selected as the best-validated candidate for this group."
