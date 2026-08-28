"""A managed tracking server chooses where artifacts live.

Databricks rejects every artifact location a client could name at
`create_experiment`: a schemeless local path ("It should have a scheme like
dbfs:/ or s3://"), a DBFS root path where root access is disabled, and even
its own `dbfs:/databricks/mlflow-tracking` prefix — which it answers by
telling the caller to leave the field unset. Since the engine's default is
the local `./mlruns`, sending it failed the whole run at pipeline begin.

A configured cloud URI must still be honoured exactly, so these pin both
halves of that rule.
"""

from __future__ import annotations

import pytest

from forecast_engine.s12_tracking.mlflow_client import _remote_artifact_location


@pytest.mark.parametrize(
    "configured",
    [
        "dbfs:/Volumes/forecastiq/forecasting/artifacts_files/mlflow",
        "abfss://container@account.dfs.core.windows.net/mlflow",
        "s3://bucket/mlflow",
        "wasbs://container@account.blob.core.windows.net/mlflow",
    ],
)
def test_a_configured_cloud_uri_is_passed_through_unchanged(configured):
    assert _remote_artifact_location(configured) == configured


@pytest.mark.parametrize(
    "configured",
    [
        "./mlruns",  # the engine's own local default
        "mlruns",
        "/var/lib/mlruns",
        "../artifacts",
        "",
        None,
    ],
)
def test_a_local_path_is_dropped_so_the_server_chooses(configured):
    assert _remote_artifact_location(configured) is None
