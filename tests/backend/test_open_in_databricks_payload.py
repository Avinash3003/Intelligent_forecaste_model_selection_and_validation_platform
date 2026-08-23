"""The Databricks deep link reaches the API payloads that render it.

The unit tests in test_databricks_links.py pin the URL shape; these pin the
wiring — that `mlflow_info`'s real `experiment_id`/`run_id` are the values
used, and that an unavailable link never degrades the rest of the MLflow
record (the whole point of making the field optional).
"""

from __future__ import annotations

from app.config.settings import Settings
from app.services.result_service import ResultService

HOST = "https://adb-1111111111111111.1.azuredatabricks.net"

# Shape taken verbatim from a real stored run's tracking_result.
_TRACKING = {
    "run_id": "8cd6e990c0f44ceab3d91e49cd665606",
    "experiment_id": "1",
    "experiment_name": "/forecast-engine",
    "tracking_uri": "databricks",
    "status": "logged",
    "models_registered": 1,
}


class _Result:
    def __init__(self, mlflow_info):
        self.mlflow_info = mlflow_info


def _service(host):
    return ResultService(
        executor=object(),
        dataset_preview_service=object(),
        settings=Settings(databricks_host=host),
    )


def test_a_databricks_tracked_run_carries_the_deep_link():
    info = _service(HOST)._mlflow_run(_Result(dict(_TRACKING)))

    assert info.databricks_run_url == (
        f"{HOST}/ml/experiments/1/runs/8cd6e990c0f44ceab3d91e49cd665606"
    )
    # The existing fields are untouched by the addition.
    assert info.run_id == "8cd6e990c0f44ceab3d91e49cd665606"
    assert info.experiment == "/forecast-engine"
    assert info.status == "logged"
    assert info.models_registered == 1


def test_the_experiment_ID_is_used_not_the_display_name():
    """`experiment` shows the human-readable name, but the Databricks route
    addresses an experiment by id — mixing them up yields a 404."""
    url = _service(HOST)._mlflow_run(_Result(dict(_TRACKING))).databricks_run_url
    assert "/ml/experiments/1/" in url
    assert "forecast-engine" not in url


def test_no_workspace_host_configured_leaves_the_record_intact():
    info = _service(None)._mlflow_run(_Result(dict(_TRACKING)))

    assert info.databricks_run_url is None
    # Everything else still renders — the link is additive, not a gate.
    assert info.run_id == "8cd6e990c0f44ceab3d91e49cd665606"
    assert info.experiment == "/forecast-engine"
    assert info.models_registered == 1


def test_a_locally_tracked_run_reports_no_link_but_still_reports_the_run():
    local = dict(_TRACKING, tracking_uri="sqlite:///mlflow.db")
    info = _service(HOST)._mlflow_run(_Result(local))

    assert info.databricks_run_url is None
    assert info.run_id == "8cd6e990c0f44ceab3d91e49cd665606"
    assert info.tracking_uri == "sqlite:///mlflow.db"


def test_a_run_with_no_tracking_record_at_all_does_not_raise():
    info = _service(HOST)._mlflow_run(_Result(None))

    assert info.databricks_run_url is None
    assert info.run_id is None


def test_a_run_missing_the_experiment_id_reports_no_link():
    partial = dict(_TRACKING)
    partial.pop("experiment_id")
    info = _service(HOST)._mlflow_run(_Result(partial))

    assert info.databricks_run_url is None
    # The display name survives even though the id needed for the URL did not.
    assert info.experiment == "/forecast-engine"
