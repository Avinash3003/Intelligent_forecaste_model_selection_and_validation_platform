"""The platform charts from data; it must not render images to storage.

Every chart a user sees is drawn in the browser by the Results page from
the run's own JSON — actuals, forecast, confidence band, per-key selection.
Nothing anywhere reads a PNG.

So the 33 figures each run used to render (10 forecast + 10 model
comparison + 10 drift + 3 summary = 3.2 MB, measured on
dbx-run-d63178305e6c) were written and never looked at, inside the tracking
stage that took 758s of a 1092s run. They are gone, and this keeps them
gone: the tempting fix for "I want to see a chart in MLflow" is to add one
back, and it would cost every run the same time again.
"""

from __future__ import annotations

import inspect

from forecast_engine.s12_tracking import artifact_logger


def test_no_producer_writes_an_image():
    """The registry is the whole surface: log_all_artifacts runs exactly it."""
    source = inspect.getsource(artifact_logger)

    assert ".png" not in source
    assert "matplotlib" not in source
    assert "pyplot" not in source
    assert "log_figure" not in source


def test_every_artifact_is_a_readable_json_text_or_dataset_payload():
    for name, producer in artifact_logger._PRODUCERS.items():
        body = inspect.getsource(producer)
        assert ".png" not in body, f"{name} writes an image"


def test_the_data_artifacts_are_all_still_produced():
    """Removing the figures must not remove what the app actually reads."""
    assert set(artifact_logger._PRODUCERS) == {
        "pipeline_configuration",
        "curated_dataset",
        "forecast_results",
        "metrics_summary",
        "shap_outputs",
        "drift_reports",
        "ranking_results",
        "fallback_report",
        "llm_business_summary",
    }


def test_the_tracking_client_offers_no_figure_logging_at_all():
    """Nothing should be able to reintroduce this by reaching for a helper."""
    from forecast_engine.s12_tracking import mlflow_client

    assert not hasattr(mlflow_client.MLflowClient, "log_figure")


# --- the curated dataset must not need a POSIX mount -------------------


def test_the_curated_dataset_is_reached_through_the_storage_adapter():
    """Gate 2 (dbx-run-ed751d39db78) failed here: a raw `path.exists()` on
    a UC Volume raised `PermissionError [Errno 1] Operation not permitted`
    inside the container, and MLflow reported
    `logged_with_artifact_errors` — the forecast was fine, the curated
    dataset simply never reached the run."""
    import inspect

    from forecast_engine.s12_tracking import artifact_logger

    source = inspect.getsource(artifact_logger._log_curated_dataset)
    assert "storage.exists" in source
    # Comments name the old call precisely because it is what must not run,
    # so only executable lines count.
    code = [l for l in source.splitlines() if not l.strip().startswith("#")]
    assert not any("path.exists()" in l for l in code)


def test_an_unreachable_curated_dataset_is_recorded_by_reference(monkeypatch):
    """Storage being unreachable must degrade to a pointer, never crash the
    tracking stage and never be silently dropped."""
    from forecast_engine.core import storage
    from forecast_engine.s12_tracking import artifact_logger

    logged = {}

    class _Client:
        def log_dict_artifact(self, payload, path):
            logged[path] = payload

        def log_artifact_file(self, *a, **k):
            raise AssertionError("must not copy a file it cannot read")

    def _boom(path):
        raise storage.StorageError("volume unreachable")

    monkeypatch.setattr(storage, "exists", _boom)
    result = type("R", (), {"curated_dataset_uri": "/Volumes/c/s/curated_files/runs/r1/x.csv"})()

    artifact_logger._log_curated_dataset(_Client(), result, object())

    assert "dataset/curated_dataset_reference.json" in logged
