"""Reads Azure OpenAI credentials from a Databricks secret scope.

`load_azure_openai_from_scope` is the path a backend-submitted run uses: the
backend passes only the scope NAME as a task parameter, and the credentials
are read here with dbutils on the cluster running the task. That is what
makes New Job Compute and Existing Compute behave alike — a task parameter
reaches both, while a cluster environment variable exists only for a cluster
the backend creates.

`apply_azure_openai_cli_overrides` resolves {{secrets/scope/key}} passed on
the --azure-openai-* flags, which databricks.yml's dev-only
forecast_pipeline_compute job still uses. Local runs set real environment
variables directly and neither path touches them.

No value is ever logged, returned to the backend, or placed in a job
definition.
"""

from __future__ import annotations

import argparse
import logging
import os
import re

from forecast_engine.config.llm_config import (
    AZURE_OPENAI_API_KEY_ENV_VAR,
    AZURE_OPENAI_DEPLOYMENT_NAME_ENV_VAR,
    AZURE_OPENAI_ENDPOINT_ENV_VAR,
)

# Matches an UNRESOLVED `{{secrets/scope/key}}` reference — the literal
# template text, not a value.
logger = logging.getLogger(__name__)

_SECRET_TEMPLATE_RE = re.compile(r"^\{\{secrets/([^/}]+)/([^}]+)\}\}$")


class SecretResolutionError(RuntimeError):
    """A secret could not be read. Carries scope and key, never a value."""


def _read_secret(scope: str, key: str) -> str:
    """Read one secret with dbutils. The single place this process touches a value."""
    # Imported here: databricks.sdk.runtime builds a live dbutils off Databricks too.
    try:
        from databricks.sdk.runtime import dbutils
    except Exception:  # noqa: BLE001 - reported safely, never re-raised raw
        raise SecretResolutionError(
            f"Cannot read secret {key!r} from scope {scope!r}: this process is not "
            "running on Databricks compute, where dbutils is available."
        ) from None

    try:
        resolved = dbutils.secrets.get(scope=scope, key=key)
    except Exception:  # noqa: BLE001 - the cause may echo its own arguments
        # from None: the cause is discarded, so nothing it carries reaches a traceback.
        raise SecretResolutionError(
            f"Secret {key!r} could not be read from scope {scope!r}. Check the scope "
            "exists, holds that key, and that the job's identity has READ on it."
        ) from None

    if not resolved:
        raise SecretResolutionError(f"Secret {key!r} in scope {scope!r} is empty.")
    return resolved


def _resolve_databricks_secret_template(value: str) -> str:
    """Resolve one {{secrets/scope/key}} literal; anything else is returned as-is."""
    match = _SECRET_TEMPLATE_RE.match(value)
    if not match:
        return value
    return _read_secret(match.group(1), match.group(2))


# Key names inside the scope. The scope is configuration; these are its contents.
_AZURE_OPENAI_SCOPE_KEYS = {
    AZURE_OPENAI_ENDPOINT_ENV_VAR: "azure-openai-endpoint",
    AZURE_OPENAI_API_KEY_ENV_VAR: "azure-openai-api-key",
    AZURE_OPENAI_DEPLOYMENT_NAME_ENV_VAR: "azure-openai-deployment",
}


def load_azure_openai_from_scope(scope: str | None) -> None:
    """Read the Azure OpenAI credentials straight from `scope` into this process."""
    scope = (scope or "").strip()
    if not scope:
        return
    for env_var, key in _AZURE_OPENAI_SCOPE_KEYS.items():
        # A real environment variable already set wins, so local runs are untouched.
        if os.environ.get(env_var):
            continue
        try:
            os.environ[env_var] = _read_secret(scope, key)
        except SecretResolutionError as exc:
            # Insights degrade to templates with a reason; a forecast is not failed for them.
            logger.warning("LLM credentials unavailable: %s", exc)
            return


def apply_azure_openai_cli_overrides(args: argparse.Namespace) -> None:
    """Copy the --azure-openai-* flags into the environment before anything
    else runs, so LLMConfig.from_env() sees them as ordinary variables.

    A no-op when real environment variables are already set. Values are
    never logged.
    """
    if args.azure_openai_endpoint:
        os.environ[AZURE_OPENAI_ENDPOINT_ENV_VAR] = _resolve_databricks_secret_template(args.azure_openai_endpoint)
    if args.azure_openai_api_key:
        os.environ[AZURE_OPENAI_API_KEY_ENV_VAR] = _resolve_databricks_secret_template(args.azure_openai_api_key)
    if args.azure_openai_deployment:
        os.environ[AZURE_OPENAI_DEPLOYMENT_NAME_ENV_VAR] = _resolve_databricks_secret_template(
            args.azure_openai_deployment
        )
