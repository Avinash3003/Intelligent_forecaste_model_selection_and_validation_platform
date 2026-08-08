"""LLM Insight Engine (Section 6.12) — the dedicated LLM service.

Consumes exactly one `PipelineResult` and produces a `BusinessInsightReport`.
It never touches training code, never receives a live model or DataFrame,
and never influences any forecasting decision — every decision it describes
(model selection, ranking, drift validation) is already final by the time
this engine runs.

Each of the seven output sections is generated independently and isolated:
one section's Azure OpenAI failure is recorded on `section_errors` and does
not prevent the others from being generated, following the same per-item
isolation principle used throughout the pipeline (Section 5.3).
"""

from __future__ import annotations

from forecast_engine.config.llm_config import LLMConfig
from forecast_engine.s11_llm.azure_openai_service import AzureOpenAIService
from forecast_engine.s11_llm.context_formatter import render_prompt_variables
from forecast_engine.s11_llm.insight_report import BusinessInsightReport
from forecast_engine.core.pipeline_result import PipelineResult
from forecast_engine.s11_llm.prompt_library import INSIGHT_SECTIONS, PromptLibrary
from forecast_engine.utils.exceptions import LLMProviderError


class LLMInsightEngine:
    """Turns a finished pipeline execution into business-readable insight."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        service: AzureOpenAIService | None = None,
        prompt_library: PromptLibrary | None = None,
    ) -> None:
        # Store config and construct default service/prompt-library
        self._config = config or LLMConfig.default()
        self._prompts = prompt_library or PromptLibrary()
        self._service = service or AzureOpenAIService(self._config)

    # Generate every insight section for one pipeline run; never raises
    def generate(self, pipeline_result: PipelineResult) -> BusinessInsightReport:
        report = BusinessInsightReport()

        if not self._config.enabled:
            report.status = "disabled"
            return report

        report.provider = "azure_openai"
        report.model_name = self._config.deployment_name

        if not self._service.is_available():
            report.status = f"unavailable: {self._service.unavailable_reason()}"
            return report

        variables = render_prompt_variables(pipeline_result, self._config)
        system_prompt = self._prompts.system_prompt()

        succeeded = 0
        for section in INSIGHT_SECTIONS:
            try:
                user_prompt = self._prompts.render(section, variables)
                text = self._service.complete(system_prompt, user_prompt)
                setattr(report, section, text)
                succeeded += 1
            except LLMProviderError as exc:  # noqa: PERF203 - isolation matters more than loop speed here
                report.section_errors[section] = str(exc)

        report.available = succeeded > 0
        if succeeded == len(INSIGHT_SECTIONS):
            report.status = "generated"
        elif succeeded > 0:
            report.status = "partial"
        else:
            report.status = "failed"

        return report
