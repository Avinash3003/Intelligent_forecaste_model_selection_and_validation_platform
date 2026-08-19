"""Deep links from a ForecastIQ run to the same run in Databricks MLflow.

Phase 1 of the MLflow integration: navigation only. The URL is built here
rather than in the frontend for two reasons — the workspace host is already
backend configuration (`Settings.databricks_host`), and this repo
deliberately keeps environment identifiers out of the compiled bundle (the
same reason the Entra config is served from `/auth/config` instead of a
`VITE_` variable).

**Authentication is deliberately NOT handled here.** Clicking the link takes
the user to Databricks, which applies its own session/SSO. Nothing in this
module reads, stores or emits a credential; it composes a URL from three
non-sensitive identifiers. Phase 2 layers authorization on top of this
function without changing its signature.
"""

from __future__ import annotations

# Databricks' own MLflow run route. The experiment id is required — a run id
# alone does not address a run in this UI.
_RUN_PATH = "/ml/experiments/{experiment_id}/runs/{run_id}"

# Tracking URIs that mean "this run lives in Databricks". A run tracked to a
# local sqlite store (EXECUTION_MODE=local) exists only on that machine, so
# pointing at Databricks would produce a confidently wrong link rather than
# no link — worse than showing nothing.
_DATABRICKS_TRACKING_PREFIXES = ("databricks",)


def _clean(value: object) -> str | None:
    """A usable identifier, or None. Treats None, non-strings, empty and
    whitespace-only values alike so callers never have to pre-validate."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def is_databricks_tracking_uri(tracking_uri: object) -> bool:
    """Whether a tracking URI refers to a Databricks-managed store."""
    cleaned = _clean(tracking_uri)
    if cleaned is None:
        return False
    return cleaned.lower().startswith(_DATABRICKS_TRACKING_PREFIXES)


def mlflow_run_url(
    workspace_host: object,
    experiment_id: object,
    run_id: object,
    tracking_uri: object = None,
) -> str | None:
    """The canonical Databricks MLflow run URL, or None when one cannot be
    built honestly.

    Returns None — never a partial or guessed URL — when the workspace host,
    experiment id or run id is missing/blank, or when `tracking_uri` is given
    and does not name a Databricks store. Callers render the link only when
    this returns a value, so an incomplete run degrades to "no button"
    instead of a link that 404s.
    """
    host = _clean(workspace_host)
    experiment = _clean(experiment_id)
    run = _clean(run_id)

    if host is None or experiment is None or run is None:
        return None

    # A tracking URI is optional context; when supplied it must agree that
    # this run is in Databricks.
    if tracking_uri is not None and not is_databricks_tracking_uri(tracking_uri):
        return None

    # Only http(s) hosts produce a followable external link. Anything else
    # (a bare hostname, a sqlite path, a mangled value) is rejected rather
    # than coerced into something that looks valid.
    if not host.lower().startswith(("https://", "http://")):
        return None

    return host.rstrip("/") + _RUN_PATH.format(experiment_id=experiment, run_id=run)
