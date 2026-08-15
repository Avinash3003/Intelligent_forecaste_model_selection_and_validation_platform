"""Generates one structured insight per forecast group.

Per group, never one narrative for the whole run — each call is scoped to a
single group's decision, which is both the contract the dashboard needs and
what prevents a reader on group B seeing group A's text.

Each insight goes through, in order:
  1. route to a deployment tier based on decision complexity.
  2. call Azure OpenAI, capturing token and latency telemetry.
  3. validate against the schema, retrying with the validation errors fed
     back into the prompt on failure.
  4. run the deterministic grounding check against the group's own metrics.
  5. on provider failure try the fallback deployment, then the deterministic
     template path, so the run still finishes with an explanation.

Every attempt, including failed and retried ones, is recorded to the trace
store before returning.
"""

from __future__ import annotations

from forecast_engine.config.llm_config import LLMConfig
from forecast_engine.core.pipeline_result import PipelineResult
from forecast_engine.s11_llm.azure_openai_service import AzureOpenAIService
from forecast_engine.s11_llm.context_formatter import (
    group_decision_facts,
    group_grounding_metrics,
    render_group_context,
)
from forecast_engine.s11_llm.grounding import check_grounding
from forecast_engine.s11_llm.insight_report import BusinessInsightReport, GroupInsight
from forecast_engine.s11_llm.llm_trace import LLMTraceStore
from forecast_engine.s11_llm.prompt_library import STRUCTURED_INSIGHT_TEMPLATE, PromptLibrary
from forecast_engine.s11_llm.schema import InsightPayload, SchemaValidationError, parse_and_validate
from forecast_engine.s11_llm.template_fallback import build_template_insight
from forecast_engine.utils.exceptions import LLMProviderError

# Reserve headroom for the response itself: a run must not spend its whole
# token budget on prompts and have nothing left to generate output with.
_MIN_TOKENS_REMAINING_TO_ATTEMPT = 200


class LLMInsightEngine:
    """Turns one finished pipeline execution into per-group structured insight."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        service: AzureOpenAIService | None = None,
        prompt_library: PromptLibrary | None = None,
        trace_store: LLMTraceStore | None = None,
    ) -> None:
        self._config = config or LLMConfig.default()
        self._prompts = prompt_library or PromptLibrary(version=self._config.prompt_version)
        self._service = service or AzureOpenAIService(self._config)
        self._trace: LLMTraceStore | None = trace_store  # built per-run in generate()

    @property
    def trace_store(self) -> LLMTraceStore | None:
        """The full per-call trace from the most recent `generate()` call.

        `BusinessInsightReport.trace_summary` (aggregate counts only) is
        what travels in the run summary; this is the detailed record
        Section 13.4 actually asks for — one entry per attempt, including
        failed and retried ones, with per-call tokens/latency/validation/
        grounding. `None` until `generate()` has run once.
        """
        return self._trace

    # Generate every group's insight for one pipeline run; never raises
    def generate(self, pipeline_result: PipelineResult) -> BusinessInsightReport:
        report = BusinessInsightReport(prompt_version=self._prompts.version)
        trace = self._trace or LLMTraceStore(pipeline_result.run_id)
        self._trace = trace

        if not self._config.enabled:
            report.status = "disabled"
            report.trace_summary = trace.summary()
            return report

        report.provider = "azure_openai"
        report.model_name = self._config.deployment_name

        # Either provider being reachable is enough to attempt the LLM
        # path — `_call_llm` tries the primary first and only reaches the
        # fallback if the primary is unavailable or errors. Gating on the
        # primary alone here would skip a perfectly usable fallback
        # deployment whenever only the primary is down.
        llm_available = self._service.is_available() or self._service.is_available(use_fallback=True)
        if not llm_available:
            report.status = f"unavailable: {self._service.unavailable_reason()}"

        group_ids = [group.get("group_id") for group in pipeline_result.forecast_groups if group.get("group_id")]

        budget = self._config.max_tokens_per_run
        tokens_used = 0
        succeeded = 0

        for group_id in group_ids:
            budget_exhausted = budget is not None and (budget - tokens_used) < _MIN_TOKENS_REMAINING_TO_ATTEMPT
            if budget_exhausted:
                report.token_budget_exhausted = True

            insight = self._generate_one(
                pipeline_result, group_id, trace, use_llm=llm_available and not budget_exhausted
            )
            report.groups[group_id] = insight
            if insight.payload is not None:
                succeeded += 1

            tokens_used = trace.summary()["total_tokens"] or 0

        report.available = succeeded > 0
        if succeeded == len(group_ids) and group_ids:
            report.status = "generated"
        elif succeeded > 0:
            report.status = "partial"
        elif not llm_available:
            pass  # status already set to the unavailable reason above
        else:
            report.status = "failed"

        report.token_budget = budget
        report.trace_summary = trace.summary()
        return report

    # Generate (or fall back to a template for) one group's insight
    def _generate_one(
        self, pipeline_result: PipelineResult, group_id: str, trace: LLMTraceStore, *, use_llm: bool
    ) -> GroupInsight:
        facts = group_decision_facts(pipeline_result, group_id)
        metrics = group_grounding_metrics(pipeline_result, group_id)
        # Model routing (Section 13.2): a clean, single-candidate win is the
        # simple case; anything with more than one rejected candidate to
        # account for is the complex case that benefits from a stronger
        # model. Neither tier need be configured — `deployment_for` falls
        # back to the primary deployment when a tier has none set.
        tier = "complex" if len(facts["rejected_candidates"]) > 1 else "simple"

        if use_llm:
            payload, provider, prompt_status, grounding_status, retries, error = self._call_llm(
                pipeline_result, group_id, facts, metrics, tier, trace
            )
            if payload is not None:
                return GroupInsight(
                    group_id=group_id,
                    payload=payload,
                    provider=provider,
                    prompt_version=self._prompts.version,
                    validation_status=prompt_status,
                    grounding_status=grounding_status,
                    retry_count=retries,
                    error=error,
                )
            # Every LLM path (primary, fallback deployment, validation
            # retries) was exhausted without a usable payload — degrade to
            # the deterministic template rather than leaving this group
            # with no explanation at all.

        payload = build_template_insight(
            selected_model=facts["selected_model"],
            wmape=facts["wmape"],
            is_fallback=facts["is_fallback"],
            fallback_trigger=facts["fallback_trigger"],
            rejected_candidates=facts["rejected_candidates"],
            confidence_pct=facts["confidence_estimate"],
            caveats=facts["caveats"],
        )
        grounding = check_grounding(payload.concise_summary, metrics)
        return GroupInsight(
            group_id=group_id,
            payload=payload,
            provider="template",
            prompt_version=self._prompts.version,
            validation_status="passed",
            grounding_status=grounding.status,
            retry_count=0,
            error=None if use_llm else self._service.unavailable_reason(),
        )

    # Attempt the LLM path (primary, then fallback deployment), with
    # bounded validation retries on each. Returns (payload, provider,
    # validation_status, grounding_status, retry_count, error) — payload is
    # None if every avenue failed.
    def _call_llm(
        self,
        pipeline_result: PipelineResult,
        group_id: str,
        facts: dict,
        metrics: dict,
        tier: str,
        trace: LLMTraceStore,
    ) -> tuple[InsightPayload | None, str, str, str, int, str | None]:
        context = render_group_context(pipeline_result, self._config, group_id)
        base_prompt = self._prompts.render(
            STRUCTURED_INSIGHT_TEMPLATE,
            {
                "context": context,
                "max_rejections": max(len(facts["rejected_candidates"]), 1),
            },
        )
        system_prompt = self._prompts.system_prompt()

        for use_fallback in (False, True):
            if use_fallback and not self._service.is_available(use_fallback=True):
                continue
            if not use_fallback and not self._service.is_available():
                continue

            provider = "azure_openai_fallback" if use_fallback else "azure_openai"
            deployment = self._config.deployment_for(tier) if not use_fallback else self._config.fallback_deployment_name

            user_prompt = base_prompt
            last_error: str | None = None
            for attempt in range(1, self._config.max_validation_retries + 2):
                call_trace = trace.start_call(
                    group_id=group_id,
                    model_name=facts["selected_model"],
                    prompt_version=self._prompts.version,
                    deployment=deployment,
                    routing_tier=tier,
                    attempt_number=attempt,
                )
                call_trace.provider = provider
                try:
                    result = self._service.complete(
                        system_prompt, user_prompt, deployment=deployment, use_fallback=use_fallback,
                        json_mode=True,
                    )
                except LLMProviderError as exc:
                    trace.finish_call(call_trace)
                    call_trace.final_status = "provider_error"
                    call_trace.error = str(exc)
                    last_error = str(exc)
                    break  # provider-level failure: retrying the same call won't help; try the next provider

                trace.finish_call(call_trace)
                call_trace.prompt_tokens = result.prompt_tokens
                call_trace.completion_tokens = result.completion_tokens
                call_trace.total_tokens = result.total_tokens
                call_trace.estimated_cost_usd = self._estimate_cost(result, tier)

                try:
                    payload = parse_and_validate(result.text, expected_model=facts["selected_model"])
                except SchemaValidationError as exc:
                    call_trace.validation_status = "failed"
                    call_trace.validation_errors = exc.problems
                    call_trace.final_status = "validation_failed"
                    last_error = "; ".join(exc.problems)
                    if attempt <= self._config.max_validation_retries:
                        # Section 13.1: "retry mechanism with failure
                        # insight embedded back into the prompt" — the
                        # model sees exactly what it got wrong.
                        user_prompt = (
                            f"{base_prompt}\n\n---\nYour previous response failed validation for these "
                            f"reason(s): {'; '.join(exc.problems)}\nRespond again with a corrected JSON "
                            "object only, fixing every issue listed above."
                        )
                        continue
                    break

                call_trace.validation_status = "passed"
                grounding = check_grounding(payload.concise_summary, metrics)
                call_trace.grounding_status = grounding.status
                call_trace.grounding_issues = grounding.issues

                if not grounding.grounded:
                    # Section 13.1: "flag or block ungrounded output before
                    # it reaches the dashboard." A well-formed JSON response
                    # citing a number absent from this group's own metrics
                    # is not treated as success — it is blocked here, not
                    # merely labelled, so a fabricated figure can never
                    # reach the UI. Retried the same way a validation
                    # failure is, since the model may simply have rounded
                    # or misread a nearby figure and can correct it when
                    # told exactly which number was wrong.
                    call_trace.final_status = "blocked_ungrounded"
                    last_error = "Ungrounded claim(s): " + "; ".join(grounding.issues)
                    if attempt <= self._config.max_validation_retries:
                        user_prompt = (
                            f"{base_prompt}\n\n---\nYour previous response cited number(s) that do not "
                            f"match this group's actual metrics: {'; '.join(grounding.issues)}\nRespond "
                            "again, using only numbers that appear in the context above."
                        )
                        continue
                    break

                call_trace.final_status = "success"
                return payload, provider, "passed", grounding.status, attempt - 1, None

        return None, "none", "failed", "skipped", 0, last_error

    def _estimate_cost(self, result, tier: str) -> float | None:
        input_rate, output_rate = self._config.pricing_for(tier)
        if input_rate is None or output_rate is None:
            return None
        if result.prompt_tokens is None or result.completion_tokens is None:
            return None
        return (result.prompt_tokens / 1000.0) * input_rate + (result.completion_tokens / 1000.0) * output_rate
