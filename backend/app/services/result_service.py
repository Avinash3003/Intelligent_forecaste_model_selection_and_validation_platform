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
    LLMTraceSummary,
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
            llm_trace=self._llm_trace(result),
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

        # Forecast stability compares the forward forecast against this
        # group's own observed history (Section 6.10). Both are passed
        # verbatim; `compute_confidence` decides whether they are long
        # enough to score and renormalizes if not.
        group = next((g for g in result.forecast_groups if g.get("group_id") == group_id), {})
        history_values = [
            observation.get("value")
            for observation in (group.get("recent_history") or [])
            if isinstance(observation.get("value"), (int, float))
        ]
        forecast_values = [
            value for value in ((winner.get("forecast") or {}).get("values") or [])
            if isinstance(value, (int, float))
        ]

        # A fallback never runs drift validation (Section 6.9) — its
        # statistic/threshold are passed as None rather than whatever the
        # last-tried ranked candidate happened to leave in `drift`, which
        # belongs to a *different* model's attempt, not the fallback's.
        confidence = compute_confidence(
            wmape=overall.get("wmape"),
            drift_statistic=None if fallback_used else drift.get("statistic"),
            drift_threshold=None if fallback_used else drift.get("threshold_value"),
            is_fallback=fallback_used,
            forecast_values=forecast_values,
            history_values=history_values,
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
                forecast_stability=(
                    round(confidence.forecast_stability, 4)
                    if confidence.forecast_stability is not None
                    else None
                ),
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
        """Build the dashboard's narrative directly from this group's own
        structured insight (`business_insights["groups"][group_id]`).

        The engine now generates one JSON payload per group (Section 6.1
        Task 10, "Final model x key") rather than one free-text narrative
        covering every group — so there is no longer a whole-run blob to
        slice apart for a single group. This method is a direct field
        mapping, not a parser: nothing here infers structure from prose.
        """
        insights = result.business_insights or {}
        groups = insights.get("groups") or {}
        entry = groups.get(group_id) if group_id else None
        payload = (entry or {}).get("insight") if entry else None

        if not payload:
            return ExplainabilityNarrative(
                key_model_headline="No business summary was generated for this run.",
                paragraphs=[],
                available=False,
                status=(entry or {}).get("error") or insights.get("status"),
                insight=DashboardInsight(),
            )

        selected_model = payload.get("selected_model") or "—"
        summary = str(payload.get("concise_summary") or "")
        rejection_reasons = list(payload.get("rejection_reasons") or [])
        caveats = list(payload.get("caveats") or [])

        paragraphs = [summary] if summary else []
        if rejection_reasons:
            paragraphs.append("Rejected candidates: " + "; ".join(rejection_reasons))

        headline = f"{selected_model} — fallback used" if fallback_used else f"{selected_model} selected"

        return ExplainabilityNarrative(
            key_model_headline=headline,
            paragraphs=paragraphs,
            available=True,
            status=entry.get("validation_status") if entry else None,
            insight=self._dashboard_insight(summary, rejection_reasons, caveats, fallback_used),
        )

    def _dashboard_insight(
        self,
        summary: str,
        rejection_reasons: list[str],
        caveats: list[str],
        fallback_used: bool,
    ) -> DashboardInsight:
        """The three short fields the dashboard card shows.

        The engine's schema (`s11_llm/schema.py`) already caps
        `concise_summary` at 70 words before this ever runs — this is the
        belt to that suspenders, so a future schema change or a template
        fallback path can never push an unbounded string onto the
        dashboard even if the upstream cap is ever loosened.
        """
        if not summary:
            return DashboardInsight()

        capped_summary, summary_cut = _cap_words(summary, 60)
        caveat, caveat_cut = _cap_words(caveats[0], 25) if caveats else ("", False)
        key_reason, reason_cut = _cap_words(_derive_key_reason(rejection_reasons, fallback_used), 15)

        return DashboardInsight(
            summary=capped_summary or None,
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

    def _llm_trace(self, result: PipelineExecutionResult) -> LLMTraceSummary:
        insights = result.business_insights or {}
        trace = insights.get("trace_summary") or {}
        return LLMTraceSummary(
            call_count=trace.get("call_count") or 0,
            prompt_tokens=trace.get("prompt_tokens") or 0,
            completion_tokens=trace.get("completion_tokens") or 0,
            total_tokens=trace.get("total_tokens") or 0,
            average_latency_ms=trace.get("average_latency_ms"),
            estimated_cost_usd=trace.get("estimated_cost_usd"),
            retry_count=trace.get("retry_count") or 0,
            groundedness_rate=trace.get("groundedness_rate"),
            prompt_version=insights.get("prompt_version"),
            token_budget=insights.get("token_budget"),
            token_budget_exhausted=bool(insights.get("token_budget_exhausted")),
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
# The engine now hands back structured JSON per group (schema-validated and
# grounding-checked before this service ever sees it — see
# forecast_engine/s11_llm/schema.py and grounding.py), so there is no
# narrative text left to parse here. What remains is a length guardrail
# (belt-and-suspenders against the engine's own 70-word cap) and a small,
# purely mechanical "why" label — no string-sniffing of prose.


def _cap_words(text: str, limit: int) -> tuple[str, bool]:
    """Return `text` capped at `limit` words, and whether it was cut."""
    words = (text or "").split()
    if len(words) <= limit:
        return " ".join(words), False
    return " ".join(words[:limit]).rstrip(",;:") + "…", True


def _derive_key_reason(rejection_reasons: list[str], fallback_used: bool) -> str:
    """A short label for why the winner won — derived from structured
    fields, never from matching words in prose.
    """
    if fallback_used:
        return "Fallback model used"
    if rejection_reasons:
        count = len(rejection_reasons)
        return f"Outranked {count} rejected candidate{'s' if count != 1 else ''}"
    return "Highest ranked candidate"
