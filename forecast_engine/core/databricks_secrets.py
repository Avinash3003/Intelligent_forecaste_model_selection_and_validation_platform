"""Resolves {{secrets/scope/key}} in the --azure-openai-* flags.

Databricks expands that template in cluster env vars but not in job
parameters, and a wheel task has no cluster env vars to use — so the value
arrives at the CLI still in template form. dbutils.secrets.get() is the only
supported way to read the real value from running code.

Only a Databricks Jobs API python_wheel_task whose job parameters reference
a secret template hits this (currently: databricks.yml's dev-only
forecast_pipeline_compute job); local runs set real environment variables
directly, and a run submitted via jobs.submit (backend/app/orchestration/
databricks_runner.py) never passes these flags at all.
"""

from __future__ import annotations

import argparse
import os
import re

from forecast_engine.config.llm_config import (
    AZURE_OPENAI_API_KEY_ENV_VAR,
    AZURE_OPENAI_DEPLOYMENT_NAME_ENV_VAR,
    AZURE_OPENAI_ENDPOINT_ENV_VAR,
)

# Matches an UNRESOLVED `{{secrets/scope/key}}` reference — the literal
# template text, not a value.
_SECRET_TEMPLATE_RE = re.compile(r"^\{\{secrets/([^/}]+)/([^}]+)\}\}$")


def _resolve_databricks_secret_template(value: str) -> str:
    """Resolve one {{secrets/scope/key}} literal. Anything not in that exact
    shape is returned untouched."""
    match = _SECRET_TEMPLATE_RE.match(value)
    if not match:
        return value
    scope, key = match.group(1), match.group(2)
    # Deliberately local: `databricks.sdk.runtime` constructs a live
    # `dbutils` at import time, which fails outside real Databricks
    # compute — a module-level import here would break every local run,
    # every local run, and every test that imports this module.
    from databricks.sdk.runtime import dbutils

    return dbutils.secrets.get(scope=scope, key=key)


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
