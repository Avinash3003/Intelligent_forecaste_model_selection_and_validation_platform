"""Artifact logging (Section 6.13, "Log Artifacts").

Every artifact is produced by its own small function, registered in
`_PRODUCERS` below. Adding a ninth artifact later — a new report, a new
plot — is registering one more function here; `log_all_artifacts` and the
tracking pipeline that calls it never change. Each producer is isolated:
one artifact failing (a plotting error, a missing curated-dataset file) is
recorded and skipped, never allowed to block the rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from forecast_engine.config.mlflow_config import MLflowConfig
from forecast_engine.core.pipeline_result import PipelineResult
from forecast_engine.s12_tracking.mlflow_client import MLflowClient
from forecast_engine.utils.exceptions import MLflowTrackingError

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
    path = Path(result.curated_dataset_uri)
    if not path.exists():
        # A non-local (e.g. cloud) curated_dataset_uri is recorded by
        # reference instead of copied — logging a pointer is still useful
        # provenance without requiring this layer to understand every
        # possible storage backend.
        client.log_dict_artifact({"curated_dataset_uri": result.curated_dataset_uri}, "dataset/curated_dataset_reference.json")
        return
    client.log_artifact_file(str(path), artifact_path="dataset")


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


# Log a forecast plot per group as a matplotlib figure
def _log_forecast_plots(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    if not result.forecast_outputs:
        return

    import matplotlib

    matplotlib.use("Agg")  # headless: no display server is available or required
    import matplotlib.pyplot as plt

    for output in _limit_groups(result.forecast_outputs, config):
        forecast = output.get("forecast")
        if not forecast or not forecast.get("values"):
            continue

        figure, axis = plt.subplots(figsize=(8, 4))
        x = range(len(forecast["values"]))
        axis.plot(x, forecast["values"], marker="o", label="Forecast")
        if forecast.get("lower") and forecast.get("upper"):
            axis.fill_between(x, forecast["lower"], forecast["upper"], alpha=0.2, label="Confidence Interval")
        axis.set_title(f"{output['forecast_group']} — {output.get('model_name')}")
        axis.set_xlabel("Period")
        axis.set_ylabel("Forecast Value")
        axis.legend()
        figure.tight_layout()

        group_slug = str(output["forecast_group"]).replace(" | ", "_").replace(" ", "_")
        client.log_figure(figure, f"plots/{group_slug}_forecast.png")
        plt.close(figure)


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


# ---------------------------------------------------------------------------
# Section 7 visualizations
#
# Every figure below is drawn from values the pipeline already produced — none
# of them recompute a metric or a decision. They exist so the MLflow UI can be
# reviewed on its own, without the platform's dashboard alongside it.
#
# All share one guard: a plotting failure must never fail the run, so each
# returns early on missing data and `log_all_artifacts` catches the rest.
# ---------------------------------------------------------------------------


def _pyplot():
    """Matplotlib in headless mode, imported lazily.

    Kept out of module import so a deployment without matplotlib still logs
    every JSON artifact rather than failing at import time.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display server on a job cluster
    import matplotlib.pyplot as plt

    return plt


def _limit_groups(items: list, config: MLflowConfig) -> list:
    """Cap a per-group figure set at `config.max_plot_groups`.

    Returns the items unchanged when the cap is zero or negative, which is how
    a deployment asks for the complete set.
    """
    cap = getattr(config, "max_plot_groups", 0)
    return items if cap <= 0 else items[:cap]


def _slug(group_id: object) -> str:
    return str(group_id).replace(" | ", "_").replace(" ", "_").replace("/", "-")


# Accuracy per key, sorted — the run-level "did this work" chart
def _log_accuracy_summary(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    # WMAPE is held in the backtest report, not on the winner record (which
    # carries the forecast itself), so the two are joined on (group, model).
    wmape_by_pair = {
        (str(row.get("group_id")), row.get("model_name")): ((row.get("backtest") or {}).get("overall") or {}).get("wmape")
        for row in (result.backtesting_metrics or {}).get("results", [])
    }
    rows = []
    for winner in result.final_winner_models:
        group = winner.get("forecast_group")
        wmape = wmape_by_pair.get((str(group), winner.get("final_production_model")))
        if group and isinstance(wmape, (int, float)):
            # Platform accuracy is 100 − WMAPE (Section 10).
            rows.append((str(group), max(0.0, 100.0 - wmape), bool(winner.get("fallback_flag"))))
    if not rows:
        return

    rows.sort(key=lambda r: r[1])
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9, max(3, 0.38 * len(rows))))
    # Fallback keys are coloured apart: their accuracy is not evidence the
    # platform chose well, only that a baseline was available.
    colors = ["#f59e0b" if fb else "#10b981" if acc >= 70 else "#ef4444" for _, acc, fb in rows]
    axis.barh([r[0] for r in rows], [r[1] for r in rows], color=colors)
    axis.axvline(70, linestyle="--", linewidth=1, color="#475569", label="70% target")
    axis.set_xlabel("Accuracy (100 − WMAPE, %)")
    axis.set_title("Forecast accuracy by key")
    axis.set_xlim(0, 100)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    client.log_figure(figure, "plots/summary_accuracy_by_key.png")
    plt.close(figure)


# WMAPE per model within each key — why the winner won, as a picture
def _log_model_comparison_plots(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    by_group: dict[str, list[tuple[str, float]]] = {}
    for row in (result.backtesting_metrics or {}).get("results", []):
        wmape = ((row.get("backtest") or {}).get("overall") or {}).get("wmape")
        if row.get("group_id") and row.get("model_name") and isinstance(wmape, (int, float)):
            by_group.setdefault(str(row["group_id"]), []).append((str(row["model_name"]), wmape))
    if not by_group:
        return

    winners = {
        str(w.get("forecast_group")): w.get("final_production_model")
        for w in result.final_winner_models
    }
    plt = _pyplot()
    for group, entries in _limit_groups(list(by_group.items()), config):
        entries.sort(key=lambda e: e[1])
        figure, axis = plt.subplots(figsize=(7, 3.2))
        # The selected model is highlighted so the chart answers "which one
        # shipped" as well as "which scored best" — they are not always the
        # same, because ranking also weighs stability and SHAP.
        colors = ["#4f46e5" if m == winners.get(group) else "#cbd5e1" for m, _ in entries]
        axis.bar([m for m, _ in entries], [v for _, v in entries], color=colors)
        axis.set_ylabel("WMAPE (%)  — lower is better")
        axis.set_title(f"Model comparison — {group}")
        figure.tight_layout()
        client.log_figure(figure, f"plots/{_slug(group)}_model_comparison.png")
        plt.close(figure)


# Historical vs forecast distribution, with the dynamic threshold marked
def _log_drift_plots(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    histories = {
        str(g.get("group_id")): (g.get("recent_history") or [])
        for g in result.forecast_groups
    }
    if not histories:
        return

    plt = _pyplot()
    for winner in _limit_groups(result.final_winner_models, config):
        group = str(winner.get("forecast_group"))
        forecast = (winner.get("forecast") or {}).get("values") or []
        history = [h.get("value") if isinstance(h, dict) else h for h in histories.get(group, [])]
        history = [v for v in history if isinstance(v, (int, float))]
        if not forecast or not history:
            continue

        figure, axis = plt.subplots(figsize=(7, 3.2))
        axis.hist(history, bins=12, alpha=0.6, label="Historical", color="#0f172a")
        axis.hist(forecast, bins=12, alpha=0.6, label="Forecast", color="#4f46e5")
        statistic = winner.get("drift_statistic")
        threshold = winner.get("dynamic_threshold_value")
        subtitle = ""
        if isinstance(statistic, (int, float)) and isinstance(threshold, (int, float)):
            verdict = "passed" if statistic <= threshold else "failed"
            subtitle = (
                f"\n{winner.get('selected_drift_algorithm')}: "
                f"{statistic:.4f} vs {threshold:.4f} threshold ({verdict})"
            )
        axis.set_title(f"Distribution overlay — {group}{subtitle}", fontsize=10)
        axis.set_xlabel("Target value")
        axis.legend(fontsize=8)
        figure.tight_layout()
        client.log_figure(figure, f"plots/{_slug(group)}_drift_distribution.png")
        plt.close(figure)


# Ranking components per model — backtest vs stability vs SHAP
def _log_ranking_heatmap(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    rankings = (result.ranking_results or {}).get("rankings") or {}
    labels: list[str] = []
    matrix: list[list[float]] = []
    for group, candidates in rankings.items():
        for candidate in candidates or []:
            scores = [
                (candidate.get("backtest") or {}).get("score"),
                (candidate.get("stability") or {}).get("score"),
                (candidate.get("shap") or {}).get("score"),
                candidate.get("composite_score"),
            ]
            if all(isinstance(v, (int, float)) for v in scores):
                labels.append(f"{group} · {candidate.get('model_name')}")
                matrix.append([float(v) for v in scores])
    if not matrix:
        return

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6.5, max(3, 0.32 * len(matrix))))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(4), ["Backtest", "Stability", "SHAP", "Composite"], fontsize=8)
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    axis.set_title("Ranking components (min-max normalized within each key)", fontsize=10)
    figure.colorbar(image, ax=axis, shrink=0.7)
    figure.tight_layout()
    client.log_figure(figure, "plots/summary_ranking_heatmap.png")
    plt.close(figure)


# Backtest metric per rolling window — is accuracy stable over time?
def _log_backtest_trend(client: MLflowClient, result: PipelineResult, config: MLflowConfig) -> None:
    winners = {
        str(w.get("forecast_group")): w.get("final_production_model")
        for w in result.final_winner_models
    }
    series: list[tuple[str, list[float]]] = []
    for row in (result.backtesting_metrics or {}).get("results", []):
        group, model = str(row.get("group_id")), row.get("model_name")
        if winners.get(group) != model:
            continue  # only the shipped model; every candidate would be unreadable
        values = [
            (w.get("metrics") or {}).get("wmape")
            for w in ((row.get("backtest") or {}).get("windows") or [])
        ]
        values = [v for v in values if isinstance(v, (int, float))]
        if len(values) > 1:
            series.append((f"{group} · {model}", values))
    if not series:
        return

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 4))
    for label, values in series:
        axis.plot(range(1, len(values) + 1), values, marker="o", linewidth=1.2, label=label)
    axis.set_xlabel("Backtest window")
    axis.set_ylabel("WMAPE (%)")
    axis.set_title("Backtest accuracy across rolling windows (selected models)")
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    client.log_figure(figure, "plots/summary_backtest_trend.png")
    plt.close(figure)


_PRODUCERS: dict[str, ArtifactProducer] = {
    "pipeline_configuration": _log_pipeline_configuration,
    "curated_dataset": _log_curated_dataset,
    "forecast_results": _log_forecast_results,
    "metrics_summary": _log_metrics_summary,
    "shap_outputs": _log_shap_outputs,
    "forecast_plots": _log_forecast_plots,
    "drift_reports": _log_drift_reports,
    "ranking_results": _log_ranking_results,
    "fallback_report": _log_fallback_report,
    "llm_business_summary": _log_llm_business_summary,
    # Section 7 visualizations — reviewable directly in the MLflow UI.
    "accuracy_summary_plot": _log_accuracy_summary,
    "model_comparison_plots": _log_model_comparison_plots,
    "drift_distribution_plots": _log_drift_plots,
    "ranking_heatmap_plot": _log_ranking_heatmap,
    "backtest_trend_plot": _log_backtest_trend,
}
