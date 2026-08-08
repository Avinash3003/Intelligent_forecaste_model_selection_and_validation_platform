"""Forecast Insights Dashboard payload (Sections 5.4–5.7).

Reads one run's real `PipelineExecutionResult` from the Pipeline Executor
and reshapes it into what the Results page renders. Nothing here recomputes
a forecasting decision: every model name, metric, drift statistic and
narrative already exists in the pipeline's own reports — this module only
selects the requested group and flattens it for display.

`_evaluated_models` is the transparency panel's data source: one row per
model actually trained for the selected group, built by joining the
training, evaluation, ranking and selection reports on (group_id,
model_name) — the same join key every one of those reports already uses.
Nothing is computed here that the pipeline did not already produce; this
module only assembles what exists into one place per model.
"""

from __future__ import annotations

import re
from typing import Any

from app.orchestration.exceptions import RunNotReadyError
from app.orchestration.executor import PipelineExecutor, get_pipeline_executor
from app.orchestration.schemas import JobStatus, PipelineExecutionResult
from app.schemas.results import (
    BacktestMetrics,
    ConfidenceExplanation,
    DriftDetail,
    EvaluatedModelDetail,
    DashboardInsight,
    ExplainabilityNarrative,
    ForecastPoint,
    ForwardValidationRule,
    GroupOption,
    MLflowRunInfo,
    ModelDecision,
    RankingBreakdown,
    ResultsResponse,
    ShapDriver,
    UnderlyingMetrics,
)
from app.services.confidence import compute_confidence

MAX_SHAP_DRIVERS = 8


class ResultService:
    """Builds the dashboard payload for one run and one forecasting group."""

    def __init__(self, executor: PipelineExecutor | None = None) -> None:
        self._executor = executor or get_pipeline_executor()

    def get_results(self, run_id: str, group_id: str | None = None) -> ResultsResponse:
        """Assemble the Results payload.

        Raises:
            UnknownRunError / RunNotReadyError: propagated from the
                executor so the route can map them to 404 / 409 — a run
                that has not finished has no results to show, and saying so
                is more honest than returning an empty dashboard.
        """
        result = self._executor.get_result(run_id)

        # A Runner returns a result envelope for a failed/cancelled run too,
        # carrying the error rather than any forecast. Building a dashboard
        # from it would render a page of "—" placeholders that looks like a
        # real but empty result, so this refuses instead — the same reason
        # a still-running run is refused above.
        if result.job_status is not JobStatus.COMPLETED:
            detail = result.error or f"Run '{run_id}' finished as {result.job_status.value}."
            raise RunNotReadyError(
                f"Run '{run_id}' has no results to show ({result.job_status.value}). {detail}"
            )

        groups = self._group_options(result)
        selected = self._resolve_group(result, group_id, groups)

        winner = (result.winner_model or {}).get(selected, {}) if selected else {}
        drift = (result.drift_results or {}).get(selected, {}) if selected else {}
        ranked = self._ranked_models(result, selected)

        return ResultsResponse(
            run_id=run_id,
            group_id=selected,
            groups=groups,
            horizon_points=self._horizon_points(winner),
            model_decision=self._model_decision(result, selected, winner, drift),
            evaluated_models=self._evaluated_models(result, selected, ranked, winner, drift),
            actual_vs_forecast=self._actual_vs_forecast(result, selected, winner),
            explainability=self._explainability(result, bool(winner.get("fallback_used")), selected),
            shap_drivers=self._shap_drivers(result, selected, winner),
            underlying_metrics=self._underlying_metrics(result, selected, winner, drift),
            mlflow_run=self._mlflow_run(result),
        )

    # ------------------------------------------------------------------
    # Group selection
    # ------------------------------------------------------------------

    def _group_options(self, result: PipelineExecutionResult) -> list[GroupOption]:
        return [
            GroupOption(
                group_id=group.get("group_id", ""),
                label=self._group_label(group),
            )
            for group in result.forecast_groups
            if group.get("group_id")
        ]

    def _group_label(self, group: dict[str, Any]) -> str:
        # Multi-series runs read better as "store=1 · item=3" than as a
        # composite id; single-series runs have no key values at all.
        key_values = group.get("key_values") or {}
        if not key_values:
            return "All data (single series)"
        return " · ".join(f"{name}={value}" for name, value in key_values.items())

    def _resolve_group(
        self, result: PipelineExecutionResult, requested: str | None, groups: list[GroupOption]
    ) -> str | None:
        available = {option.group_id for option in groups}
        if requested and requested in available:
            return requested
        return groups[0].group_id if groups else None

    # ------------------------------------------------------------------
    # Model Decision panel (Section 5.5) + confidence
    # ------------------------------------------------------------------

    def _model_decision(
        self,
        result: PipelineExecutionResult,
        group_id: str | None,
        winner: dict[str, Any],
        drift: dict[str, Any],
    ) -> ModelDecision:
        model_name = winner.get("model_name") or "—"
        fallback_used = bool(winner.get("fallback_used"))

        overall = self._winning_backtest_metrics(result, group_id, model_name)
        drift_result = drift.get("result") or {}

        # A fallback never runs drift validation (Section 6.9) — its
        # statistic/threshold are passed as None rather than whatever the
        # last-tried ranked candidate happened to leave in `drift`, which
        # belongs to a *different* model's attempt, not the fallback's.
        confidence = compute_confidence(
            wmape=overall.get("wmape"),
            drift_statistic=None if fallback_used else drift.get("statistic"),
            drift_threshold=None if fallback_used else drift.get("threshold_value"),
            is_fallback=fallback_used,
        )
        # `drift_result` is still read for the underlying-metrics panel's
        # verbatim "Passed"/"Not Passed" label — unrelated to confidence.
        _ = drift_result

        return ModelDecision(
            selected_model=model_name,
            confidence=round(confidence.confidence, 4) if confidence.confidence is not None else None,
            confidence_explanation=ConfidenceExplanation(
                backtest_accuracy=(
                    round(confidence.backtest_accuracy, 4) if confidence.backtest_accuracy is not None else None
                ),
                drift_margin=round(confidence.drift_margin, 4) if confidence.drift_margin is not None else None,
                formula=confidence.formula,
                explanation=confidence.explanation,
            ),
            ranking_position=int(winner.get("final_rank") or 0),
            validation_status=winner.get("final_selection_status") or "Unknown",
            fallback_used=fallback_used,
            fallback_reason=winner.get("fallback_trigger") if fallback_used else None,
            original_candidates=list(winner.get("original_candidates") or []),
        )

    # ------------------------------------------------------------------
    # Per-model transparency panel
    # ------------------------------------------------------------------

    def _evaluated_models(
        self,
        result: PipelineExecutionResult,
        group_id: str | None,
        ranked: list[dict[str, Any]],
        winner: dict[str, Any],
        drift: dict[str, Any],
    ) -> list[EvaluatedModelDetail]:
        if not group_id:
            return []

        training_by_model = {
            entry.get("model_name"): entry
            for entry in (result.metrics or {}).get("training", {}).get("results", [])
            if entry.get("group_id") == group_id
        }
        evaluation_by_model = {
            entry.get("model_name"): entry
            for entry in (result.forecast_results or {}).get("results", [])
            if entry.get("group_id") == group_id
        }
        ranking_by_model = {entry.get("model_name"): entry for entry in ranked}

        winning_name = winner.get("model_name")
        is_fallback_winner = bool(winner.get("fallback_used"))
        # Candidates the sequential Final Selection loop actually tried and
        # rejected (Section 6.9) — every one of these reached drift
        # validation and failed it, or had no forecast to validate at all.
        rejected_reasons = {
            entry.get("model_name"): entry.get("reason")
            for entry in (winner.get("failure_reasons") or [])
        }

        model_names = set(training_by_model) | set(evaluation_by_model) | set(ranking_by_model)
        if winning_name:
            model_names.add(winning_name)

        return [
            self._one_evaluated_model(
                name,
                training_by_model.get(name, {}),
                evaluation_by_model.get(name, {}),
                ranking_by_model.get(name),
                winning_name,
                is_fallback_winner,
                rejected_reasons,
                drift,
            )
            for name in sorted(model_names)
        ]

    def _one_evaluated_model(
        self,
        name: str,
        training: dict[str, Any],
        evaluation: dict[str, Any],
        ranking_entry: dict[str, Any] | None,
        winning_name: str | None,
        is_fallback_winner: bool,
        rejected_reasons: dict[str, Any],
        drift: dict[str, Any],
    ) -> EvaluatedModelDetail:
        is_winner = name == winning_name
        is_fallback_row = is_winner and is_fallback_winner

        # The fallback baseline (e.g. seasonal_naive) is deliberately
        # excluded from the normal candidate registry (`enabled=False` in
        # model_config.py) — it never goes through Train Models or Evaluate
        # Models, and is trained fresh, on demand, only inside Final
        # Selection's fallback path (production_selector.py:_use_fallback).
        # So a fallback row genuinely has no training_report/evaluation_report
        # entry to read — this is architecture, not a missing-data bug.
        training_status = training.get("status") or (
            "Trained (fallback path — outside the normal candidate registry)" if is_fallback_row else "Not Attempted"
        )
        forward_status = evaluation.get("status") or (
            "Not applicable — the fallback baseline is never backtested or forward-validated"
            if is_fallback_row
            else "—"
        )

        return EvaluatedModelDetail(
            model=name,
            training_status=training_status,
            training_error=training.get("error"),
            backtest=self._backtest_metrics(evaluation),
            forward_validation_status=forward_status,
            forward_validation_reasons=list(evaluation.get("rejection_reasons") or []),
            forward_validation_rules=self._forward_validation_rules(evaluation),
            ranking=self._ranking_breakdown(ranking_entry),
            drift=self._drift_detail(name, winning_name, is_fallback_winner, drift, rejected_reasons),
            selection_outcome=self._selection_outcome(
                name, winning_name, is_fallback_winner, training, evaluation, rejected_reasons, ranking_entry
            ),
        )

    def _backtest_metrics(self, evaluation: dict[str, Any]) -> BacktestMetrics | None:
        backtest = evaluation.get("backtest") or {}
        overall = backtest.get("overall") or {}
        if not overall:
            return None
        return BacktestMetrics(
            wmape=overall.get("wmape"),
            rmse=overall.get("rmse"),
            mae=overall.get("mae"),
            mape=overall.get("mape"),
            smape=overall.get("smape"),
            window_count=backtest.get("window_count") or 0,
        )

    def _forward_validation_rules(self, evaluation: dict[str, Any]) -> list[ForwardValidationRule]:
        outcomes = ((evaluation.get("validation") or {}).get("rule_outcomes")) or []
        return [
            ForwardValidationRule(
                rule_name=outcome.get("rule_name", "—"),
                passed=bool(outcome.get("passed")),
                detail=outcome.get("detail"),
            )
            for outcome in outcomes
        ]

    def _ranking_breakdown(self, entry: dict[str, Any] | None) -> RankingBreakdown | None:
        if not entry:
            return None
        backtest = entry.get("backtest") or {}
        stability = entry.get("stability") or {}
        shap = entry.get("shap") or {}
        return RankingBreakdown(
            composite_score=entry.get("composite_score"),
            backtest_score=backtest.get("score"),
            stability_score=stability.get("score"),
            shap_score=shap.get("score"),
            shap_method=shap.get("method"),
            original_backtest_rank=entry.get("original_backtest_rank"),
            final_rank=entry.get("final_composite_rank"),
        )

    def _drift_detail(
        self,
        name: str,
        winning_name: str | None,
        is_fallback_winner: bool,
        drift: dict[str, Any],
        rejected_reasons: dict[str, Any],
    ) -> DriftDetail | None:
        if name == winning_name:
            if is_fallback_winner:
                return DriftDetail(evaluated=False, detail="The fallback path bypasses drift validation entirely.")
            result = drift.get("result") or {}
            return DriftDetail(
                evaluated=True,
                algorithm=drift.get("algorithm"),
                statistic=drift.get("statistic"),
                threshold_value=drift.get("threshold_value"),
                threshold_method=drift.get("threshold_method"),
                passed=result.get("passed"),
                detail=result.get("detail"),
            )

        if name in rejected_reasons:
            reason = str(rejected_reasons[name])
            # The reason text itself carries the real statistic/threshold
            # (drift_validator.py formats them into `detail`) — only
            # candidates that actually reached the test mention one; a
            # candidate with no forecast to validate does not.
            return DriftDetail(evaluated="statistic" in reason.lower(), passed=False, detail=reason)

        return None  # never reached — see `selection_outcome` for why

    def _selection_outcome(
        self,
        name: str,
        winning_name: str | None,
        is_fallback_winner: bool,
        training: dict[str, Any],
        evaluation: dict[str, Any],
        rejected_reasons: dict[str, Any],
        ranking_entry: dict[str, Any] | None,
    ) -> str:
        if name == winning_name:
            return "Fallback Used" if is_fallback_winner else "Selected"

        if name in rejected_reasons:
            return f"Rejected — {rejected_reasons[name]}"

        training_status = training.get("status")
        if training_status and training_status != "Trained":
            return f"Failed to train — {training.get('error') or training_status}"

        eval_status = evaluation.get("status")
        if eval_status == "Eliminated":
            reasons = evaluation.get("rejection_reasons") or []
            return f"Eliminated — {', '.join(reasons)}" if reasons else "Eliminated at forward validation"
        if eval_status == "Failed":
            return f"Evaluation failed — {evaluation.get('error') or 'unknown error'}"
        if eval_status == "Skipped":
            return f"Skipped — {(evaluation.get('backtest') or {}).get('skipped_reason') or 'not evaluated'}"

        if ranking_entry:
            return "Not reached — a higher-ranked candidate already passed drift validation"

        return "Not evaluated"

    # ------------------------------------------------------------------
    # Other panels
    # ------------------------------------------------------------------

    def _actual_vs_forecast(
        self, result: PipelineExecutionResult, group_id: str | None, winner: dict[str, Any]
    ) -> list[ForecastPoint]:
        points: list[ForecastPoint] = []

        group = next(
            (g for g in result.forecast_groups if g.get("group_id") == group_id),
            {},
        )
        for observation in group.get("recent_history") or []:
            points.append(
                ForecastPoint(period=self._short_date(observation.get("date")), actual=observation.get("value"))
            )

        forecast = winner.get("forecast") or {}
        dates = forecast.get("dates") or []
        values = forecast.get("values") or []
        lower = forecast.get("lower") or []
        upper = forecast.get("upper") or []

        # Join the two lines by repeating the last actual as the forecast's
        # first point, so the chart draws one continuous series.
        if points and values:
            points[-1].forecast = points[-1].actual

        for index, value in enumerate(values):
            points.append(
                ForecastPoint(
                    period=f"T{index + 1}",
                    forecast=value,
                    lower=lower[index] if index < len(lower) else None,
                    upper=upper[index] if index < len(upper) else None,
                )
            )

        return points

    def _horizon_points(self, winner: dict[str, Any]) -> list[str]:
        values = (winner.get("forecast") or {}).get("values") or []
        return [f"T{index + 1}" for index in range(len(values))]

    def _explainability(
        self,
        result: PipelineExecutionResult,
        fallback_used: bool = False,
        group_id: str | None = None,
    ) -> ExplainabilityNarrative:
        insights = result.business_insights or {}
        available = bool(insights.get("available"))

        # The engine writes one narrative covering every forecast group, but
        # this dashboard shows exactly one. Rendering the whole thing meant a
        # reader scrolled past nine other groups' decisions to find theirs, so
        # the winner summary is narrowed to the selected group first. The
        # business explanation is genuinely run-level and is kept whole.
        winner_summary = _group_section(insights.get("winner_model_summary"), group_id)
        paragraphs = [text for text in (winner_summary, insights.get("business_explanation")) if text]

        return ExplainabilityNarrative(
            key_model_headline=insights.get("forecast_summary") or "No business summary was generated for this run.",
            paragraphs=paragraphs,
            available=available,
            status=insights.get("status"),
            insight=self._dashboard_insight(insights, paragraphs, fallback_used, winner_summary),
        )

    def _dashboard_insight(
        self,
        insights: dict[str, Any],
        paragraphs: list[str],
        fallback_used: bool = False,
        winner_summary: str = "",
    ) -> DashboardInsight:
        """Compress the narrative into the three short fields the dashboard shows.

        This is a guardrail, not a formatter: the caps are enforced here so an
        unusually long (or unusually structured) model response can never push a
        multi-hundred-word paragraph onto the dashboard. The engine's prompt asks
        for brevity, but a prompt is a request — this is the enforcement.
        """
        if not insights.get("available"):
            return DashboardInsight()

        source = (winner_summary or "").strip()
        if not source and paragraphs:
            source = paragraphs[0].strip()
        if not source:
            return DashboardInsight()

        sentences = _split_sentences(source)
        # Sentences that are pure field echoes ("Selected model: ARIMA") are
        # dropped: the dashboard already shows those as their own tiles, and
        # repeating them here is exactly the duplication this card exists to
        # remove.
        prose = [s for s in sentences if not re.match(r"^(selected model|model)\s*:", s, re.I)] or sentences
        summary, summary_cut = _cap_words(" ".join(prose[:2]), 60)

        caveat, caveat_cut = _cap_words(_pick_caveat(sentences), 25)
        key_reason, reason_cut = _cap_words(_derive_key_reason(source, fallback_used), 15)

        return DashboardInsight(
            summary=summary or None,
            key_reason=key_reason or None,
            caveat=caveat or None,
            truncated=summary_cut or caveat_cut or reason_cut,
        )

    def _shap_drivers(
        self, result: PipelineExecutionResult, group_id: str | None, winner: dict[str, Any]
    ) -> list[ShapDriver]:
        model_name = winner.get("model_name")
        if not group_id or not model_name:
            return []

        for entry in (result.explainability or {}).get("results", []):
            if entry.get("group_id") == group_id and entry.get("model_name") == model_name:
                importances = entry.get("importances") or {}
                ranked = sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True)
                return [
                    ShapDriver(feature=name, importance=round(float(value), 4))
                    for name, value in ranked[:MAX_SHAP_DRIVERS]
                ]
        return []

    def _underlying_metrics(
        self,
        result: PipelineExecutionResult,
        group_id: str | None,
        winner: dict[str, Any],
        drift: dict[str, Any],
    ) -> UnderlyingMetrics:
        overall = self._winning_backtest_metrics(result, group_id, winner.get("model_name"))
        drift_result = drift.get("result") or {}

        return UnderlyingMetrics(
            drift_test=str(drift.get("algorithm") or "—"),
            threshold_method=str(drift.get("threshold_method") or "—"),
            threshold_value=self._number(drift.get("threshold_value")),
            drift_score=self._number(drift.get("statistic")),
            wmape=self._percent(overall.get("wmape")),
            rmse=self._number(overall.get("rmse")),
            mae=self._number(overall.get("mae")),
            validation_result="Passed" if drift_result.get("passed") else "Not Passed",
        )

    def _winning_backtest_metrics(
        self, result: PipelineExecutionResult, group_id: str | None, model_name: str | None
    ) -> dict[str, Any]:
        for entry in (result.forecast_results or {}).get("results", []):
            if entry.get("group_id") == group_id and entry.get("model_name") == model_name:
                return ((entry.get("backtest") or {}).get("overall")) or {}
        return {}

    def _ranked_models(self, result: PipelineExecutionResult, group_id: str | None) -> list[dict[str, Any]]:
        rankings = ((result.metrics or {}).get("ranking") or {}).get("rankings") or {}
        return rankings.get(group_id, []) if group_id else []

    def _mlflow_run(self, result: PipelineExecutionResult) -> MLflowRunInfo:
        info = result.mlflow_info or {}
        return MLflowRunInfo(
            run_id=info.get("run_id"),
            experiment=info.get("experiment_name") or info.get("experiment_id"),
            status=info.get("status"),
            tracking_uri=info.get("tracking_uri"),
            models_registered=info.get("models_registered"),
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _short_date(self, value: Any) -> str:
        # "2024-03-01T00:00:00" -> "2024-03"; the chart axis is monthly.
        text = str(value or "")
        return text[:7] if len(text) >= 7 else text

    def _number(self, value: Any) -> str:
        return f"{float(value):.4g}" if isinstance(value, (int, float)) else "—"

    def _percent(self, value: Any) -> str:
        return f"{float(value):.2f}%" if isinstance(value, (int, float)) else "—"


# --- dashboard insight helpers -------------------------------------------------
# Kept module-level and pure so the length caps are trivially testable and are
# applied identically no matter which narrative section they are fed.

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# The engine's narrative is markdown-ish: group headers, bullets, bold runs.
# Those read as noise once the text is placed in a dashboard card, so they are
# stripped before the text is ever measured or capped.
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")
_GROUP_HEADER = re.compile(r"^\s*[\w\s|]{0,40}?\d+\s*[:|]\s*")
_MD_EMPHASIS = re.compile(r"[*_`#]+")
# A sentence that merely reports the absence of a problem is not a caveat.
_NEGATED = re.compile(r"\b(no|not|none|without|never)\b", re.IGNORECASE)


def _clean(text: str) -> str:
    """Flatten one narrative line into plain prose."""
    line = _MD_EMPHASIS.sub("", text or "")
    line = _GROUP_HEADER.sub("", line)
    line = _LIST_MARKER.sub("", line)
    return line.strip(" -\u2013\u2014\t")


def _split_sentences(text: str) -> list[str]:
    """Sentences, with markdown structure flattened away first.

    Bulleted lines rarely end in a full stop, so each line is treated as its own
    sentence before the usual punctuation split — otherwise an entire bullet
    list collapses into one run-on "sentence".
    """
    sentences: list[str] = []
    for raw_line in (text or "").splitlines():
        line = _clean(raw_line)
        if not line:
            continue
        for part in _SENTENCE_END.split(line):
            part = part.strip()
            # A fragment with no letters (a stray bullet, a lone full stop
            # left behind by marker stripping) is punctuation, not a sentence.
            if re.search(r"[A-Za-z]", part):
                sentences.append(part if part.endswith((".", "!", "?")) else part + ".")
    return sentences


def _cap_words(text: str, limit: int) -> tuple[str, bool]:
    """Return the text capped at `limit` words, and whether it was cut."""
    words = _clean(text).split()
    if len(words) <= limit:
        return " ".join(words), False
    return " ".join(words[:limit]).rstrip(",;:") + "\u2026", True


def _pick_caveat(sentences: list[str]) -> str:
    """The first sentence that genuinely warns about something.

    Negated forms ("No fallback model was used") match the same keywords as a
    real caveat but mean the opposite, so they are excluded rather than shown
    as a warning the run does not actually carry.
    """
    for sentence in sentences:
        lowered = sentence.lower()
        if not any(w in lowered for w in ("however", "caveat", "caution", "limited", "risk", "unstable", "short history")):
            continue
        if _NEGATED.search(lowered):
            continue
        return sentence
    return ""


def _derive_key_reason(text: str, fallback_used: bool) -> str:
    """A short label for why the winner won.

    `fallback_used` comes from the selection report rather than from string
    matching: the narrative mentions the word "fallback" even when saying one
    was *not* needed, which previously mislabelled clean runs.
    """
    lowered = (text or "").lower()
    if fallback_used:
        return "Fallback model used"
    if "drift" in lowered and "passed" in lowered:
        return "Passed drift validation"
    if "wmape" in lowered or "accuracy" in lowered:
        return "Best backtest accuracy"
    return "Highest ranked candidate"


def _group_section(narrative: str | None, group_id: str | None) -> str:
    """Return only the block of `narrative` written about `group_id`.

    The engine labels each group's section with the group id followed by a
    colon ("1 | 1:"), which is a join of the key values — meaningful to the
    pipeline, meaningless as a heading on a dashboard that already names the
    selected key. The matching block is returned with that label stripped; if
    no label is found the narrative is returned unchanged, so a single-series
    run (which has no per-group headings) still shows its text.
    """
    text = (narrative or "").strip()
    if not text or not group_id:
        return text

    # Sections start at a line that is exactly the group id plus a colon.
    pattern = re.compile(r"^\s*(.+?)\s*:\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    labelled = [m for m in matches if _looks_like_group_label(m.group(1))]
    if not labelled:
        return text

    for index, match in enumerate(labelled):
        if match.group(1).strip() != group_id.strip():
            continue
        start = match.end()
        end = labelled[index + 1].start() if index + 1 < len(labelled) else len(text)
        return text[start:end].strip()

    # The narrative exists but says nothing about this group — better to show
    # nothing than another group's decision under this group's heading.
    return ""


def _looks_like_group_label(candidate: str) -> bool:
    """Distinguish a section heading from a bullet that happens to end in a colon.

    Group ids are short and may contain spaces and separators ("1 | 1",
    "store=1 · item=1"), so length and word count identify them; a leading
    dash marks a list item, which is content rather than a heading.
    """
    text = (candidate or "").strip()
    return bool(text) and len(text) <= 80 and len(text.split()) <= 8 and not text.startswith(("-", "*", "\u2022"))
