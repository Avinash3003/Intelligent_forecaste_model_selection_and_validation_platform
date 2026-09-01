"""Logs each run artifact to MLflow.

Every artifact is produced by its own function, registered in _PRODUCERS, so
adding one is registering a function rather than changing this module's
callers. Each producer is isolated: one failing is recorded and skipped, not
allowed to block the rest.
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Callable

from forecast_engine.config.mlflow_config import MLflowConfig
from forecast_engine.core.pipeline_result import PipelineResult
from forecast_engine.core import storage
from forecast_engine.s12_tracking.mlflow_client import MLflowClient
from forecast_engine.utils.exceptions import MLflowTrackingError

logger = logging.getLogger(__name__)

ArtifactProducer = Callable[[MLflowClient, PipelineResult, MLflowConfig], None]


# Run every registered artifact producer against the active run
def log_all_artifacts(
    client: MLflowClient, pipeline_result: PipelineResult, config: MLflowConfig
) -> dict[str, str]:
    # Returns a mapping of producer name -> error message for any that failed
    errors: dict[str, str] = {}
    for name, producer in _PRODUCERS.items():
        try:
            producer(client, pipeline_result, config)
        except MLflowTrackingError as exc:
            errors[name] = str(exc)
        except Exception as exc:  # noqa: BLE001 - a plotting/serialization fault must not block other artifacts
            errors[name] = f"{type(exc).__name__}: {exc}"
    return errors


# Log the pipeline/forecast configuration as a JSON artifact
def _log_pipeline_configuration(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    client.log_dict_artifact(
        {
            "forecast_configuration": result.forecast_configuration,
            "pipeline_configuration": result.pipeline_configuration,
            "forecast_horizon": result.forecast_horizon,
            "selected_models": result.selected_models,
            "hyperparameters": result.hyperparameters,
        },
        "configuration/pipeline_configuration.json",
    )


# Log the curated dataset file, or a reference if it's non-local
def _log_curated_dataset(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    if not result.curated_dataset_uri:
        return
    uri = result.curated_dataset_uri
    path = Path(uri)

    # Asked through the storage adapter, not the filesystem. This URI is a
    # UC Volume path on every cloud run, and a container has no /Volumes
    # mount — `path.exists()` raised
    #   PermissionError: [Errno 1] Operation not permitted: '/Volumes/...'
    # on a real DCS run (dbx-run-ed751d39db78), which MLflow reported as
    # `logged_with_artifact_errors`: the forecast was fine, but the curated
    # dataset never made it onto the run. The adapter reaches the same file
    # over the Files API, and raises rather than answering False when
    # storage is unreachable.
    try:
        present = storage.exists(path)
    except Exception as exc:  # noqa: BLE001 - provenance must not fail the run
        logger.warning("Could not reach the curated dataset at %s: %s", uri, exc)
        present = False

    if not present:
        # A curated dataset this process cannot read is recorded by
        # reference instead of copied — a pointer is still useful
        # provenance without this layer understanding every backend.
        client.log_dict_artifact({"curated_dataset_uri": uri}, "dataset/curated_dataset_reference.json")
        return

    if storage.supports_atomic_replace(path):
        # A real local file: hand MLflow the path and let it stream.
        client.log_artifact_file(str(path), artifact_path="dataset")
        return

    # Reached over the Files API: MLflow needs a local file, so the bytes
    # are staged in the driver's own temp space (never in the workspace)
    # and removed immediately afterwards.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / path.name
        local.write_bytes(storage.read_bytes(path))
        client.log_artifact_file(str(local), artifact_path="dataset")


# Log the forecast outputs as a JSON artifact
def _log_forecast_results(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    client.log_dict_artifact(result.forecast_outputs, "forecast/forecast_results.json")


# Log training and backtesting metrics as a JSON artifact
def _log_metrics_summary(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    client.log_dict_artifact(
        {"training_summary": result.training_summary, "backtesting_metrics": result.backtesting_metrics},
        "metrics/metrics_summary.json",
    )


# Log SHAP explainability outputs as a JSON artifact
def _log_shap_outputs(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    client.log_dict_artifact(result.explainability_results, "explainability/shap_outputs.json")




# Log drift and threshold results as a JSON artifact
def _log_drift_reports(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    client.log_dict_artifact(
        {"drift": result.drift_results, "threshold": result.threshold_results},
        "drift/drift_report.json",
    )


# Log model ranking results as a JSON artifact
def _log_ranking_results(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    client.log_dict_artifact(result.ranking_results, "ranking/ranking_results.json")


# Log why/where the fallback model was used, per group
def _log_fallback_report(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    # Why the fallback ran, per group: trigger, candidates considered, and
    # each candidate's failure reason (Section 6.9).
    triggered = [winner for winner in result.final_winner_models if winner.get("fallback_flag")]
    client.log_dict_artifact(
        {
            "configured_fallback_model": result.fallback_model,
            "groups_using_fallback": len(triggered),
            "groups": [
                {
                    "forecast_group": winner.get("forecast_group"),
                    "fallback_model": winner.get("fallback_model"),
                    "fallback_trigger": winner.get("fallback_trigger"),
                    "original_candidates": winner.get("original_candidates"),
                    "failure_reasons": winner.get("failure_reasons"),
                }
                for winner in triggered
            ],
        },
        "selection/fallback_report.json",
    )


# Log the LLM business summary as JSON, plus a human-readable markdown doc
def _log_llm_business_summary(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    insights = result.business_insights
    client.log_dict_artifact(insights, "insights/business_summary.json")

    # Section 13.4's actual observability requirement: one record per LLM
    # call, not just the aggregate counts already inside `insights`. Logged
    # before the early return below so a run where every call failed —
    # exactly the case this exists to debug — still gets its trace.
    if result.llm_trace and result.llm_trace.get("calls"):
        client.log_dict_artifact(result.llm_trace, "insights/llm_trace.json")

    if not insights.get("available"):
        return

    # A rendered document alongside the structured JSON, since MLflow's UI
    # previews markdown/text artifacts directly — the JSON is for
    # programmatic consumers, this is for a human reading the run.
    sections = (
        ("Forecast Summary", insights.get("forecast_summary")),
        ("Winner Model Summary", insights.get("winner_model_summary")),
        ("Important Features", insights.get("important_features")),
        ("Drift Summary", insights.get("drift_summary")),
        ("Forecast Risks", insights.get("forecast_risks")),
        ("Business Explanation", insights.get("business_explanation")),
        ("Technical Explanation", insights.get("technical_explanation")),
    )
    document = "\n\n".join(f"## {title}\n\n{text}" for title, text in sections if text)
    if document:
        client.log_text_artifact(document, "insights/business_summary.md")


_PRODUCERS: dict[str, ArtifactProducer] = {
    "pipeline_configuration": _log_pipeline_configuration,
    "curated_dataset": _log_curated_dataset,
    "forecast_results": _log_forecast_results,
    "metrics_summary": _log_metrics_summary,
    "shap_outputs": _log_shap_outputs,
    "drift_reports": _log_drift_reports,
    "ranking_results": _log_ranking_results,
    "fallback_report": _log_fallback_report,
    "llm_business_summary": _log_llm_business_summary,
}
