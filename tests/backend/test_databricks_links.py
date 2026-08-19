"""The "Open in Databricks" deep link (Phase 1 — navigation only).

The contract these pin: the URL is either exactly right or absent. A
half-built or guessed URL is worse than no button, because it sends an
auditor to a 404 while looking like a working control.

Authentication is deliberately out of scope here — clicking the link lands
on Databricks, which applies its own session. Nothing in this module handles
a credential.
"""

from __future__ import annotations

import pytest

from app.services.databricks_links import is_databricks_tracking_uri, mlflow_run_url

HOST = "https://adb-1111111111111111.1.azuredatabricks.net"


# ---------------------------------------------------------------------
# 1-2. A complete run produces the canonical URL, on the configured host
# ---------------------------------------------------------------------


def test_a_complete_run_produces_the_canonical_databricks_url():
    assert mlflow_run_url(HOST, "1", "8cd6e990c0f44ceab3d91e49cd665606", "databricks") == (
        f"{HOST}/ml/experiments/1/runs/8cd6e990c0f44ceab3d91e49cd665606"
    )


def test_the_configured_workspace_host_is_the_one_used():
    """A second workspace must not inherit the first one's host — this is
    what makes the link follow configuration rather than a constant."""
    other = "https://adb-2222222222222222.2.azuredatabricks.net"
    assert mlflow_run_url(other, "42", "abc", "databricks").startswith(other)


def test_a_trailing_slash_on_the_host_does_not_double_up():
    assert mlflow_run_url(HOST + "/", "1", "abc", "databricks") == f"{HOST}/ml/experiments/1/runs/abc"


# ---------------------------------------------------------------------
# 3-5. Missing identifiers produce no link, never a broken one
# ---------------------------------------------------------------------


@pytest.mark.parametrize("run_id", [None, "", "   "])
def test_a_missing_run_id_produces_no_link(run_id):
    assert mlflow_run_url(HOST, "1", run_id, "databricks") is None


@pytest.mark.parametrize("experiment_id", [None, "", "   "])
def test_a_missing_experiment_id_produces_no_link(experiment_id):
    assert mlflow_run_url(HOST, experiment_id, "abc", "databricks") is None


@pytest.mark.parametrize("host", [None, "", "   "])
def test_a_missing_workspace_host_produces_no_link(host):
    assert mlflow_run_url(host, "1", "abc", "databricks") is None


# ---------------------------------------------------------------------
# 6. Degenerate input never raises and never yields a malformed URL
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "host,experiment,run",
    [
        (None, None, None),
        (123, "1", "abc"),          # non-string host
        (HOST, {"id": 1}, "abc"),   # non-string experiment
        (HOST, "1", ["abc"]),       # non-string run
        ("not-a-url", "1", "abc"),  # host without a scheme
        ("sqlite:///mlflow.db", "1", "abc"),
    ],
)
def test_degenerate_input_returns_none_rather_than_raising(host, experiment, run):
    assert mlflow_run_url(host, experiment, run, "databricks") is None


def test_a_locally_tracked_run_gets_no_databricks_link():
    """EXECUTION_MODE=local tracks to sqlite on that machine. The run does
    not exist in Databricks, so a link to it would be confidently wrong."""
    assert mlflow_run_url(HOST, "1", "abc", "sqlite:///mlflow.db") is None


def test_tracking_uri_is_optional_context_not_a_requirement():
    """Omitted entirely, the caller is asserting nothing about the store, so
    the link is built from the ids it did supply."""
    assert mlflow_run_url(HOST, "1", "abc") is not None


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("databricks", True),
        ("databricks-uc", True),
        ("DATABRICKS", True),
        ("  databricks  ", True),
        ("sqlite:///mlflow.db", False),
        ("", False),
        (None, False),
        (42, False),
    ],
)
def test_tracking_uri_classification(uri, expected):
    assert is_databricks_tracking_uri(uri) is expected
