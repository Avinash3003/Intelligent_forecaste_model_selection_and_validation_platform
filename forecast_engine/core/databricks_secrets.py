"""Databricks Serverless secret resolution for the AZURE_OPENAI_* CLI flags
(see `databricks/resources/forecast_job_serverless.yml`'s `business_insights`
task).

Databricks resolves `{{secrets/scope/key}}` template syntax in a handful of
fields — a cluster's `spark_env_vars` among them — but NOT in job
parameters, which is how the three Azure OpenAI values reach
`run_pipeline.py`'s `--azure-openai-*` flags on Serverless (a
`python_wheel_task` has no `spark_env_vars` equivalent to set these as real
environment variables directly). A value sourced that way arrives at the
CLI still in literal template form, and `dbutils.secrets.get()` — the only
Databricks-supported way to read an actual secret value from arbitrary
running code, since the plain Secrets REST API deliberately refuses to
return values — is what resolves it.

Local runs and the DCS job never hit this: DCS sets the real environment
variables directly via `spark_env_vars`, and a local `.env` sets them
directly too, so neither ever passes a `--azure-openai-*` flag shaped like
`{{secrets/...}}` in the first place.
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
    """Resolve one `{{secrets/scope/key}}` literal via dbutils.secrets.get().

    Available only on real Databricks compute; anything that is not this
    literal template shape (a real value, or unset) is returned/left
    untouched.
    """
    match = _SECRET_TEMPLATE_RE.match(value)
    if not match:
        return value
    scope, key = match.group(1), match.group(2)
    # Deliberately local: `databricks.sdk.runtime` constructs a live
    # `dbutils` at import time, which fails outside real Databricks
    # compute — a module-level import here would break every local run,
    # every DCS run, and every test that imports this module.
    from databricks.sdk.runtime import dbutils

    return dbutils.secrets.get(scope=scope, key=key)


def apply_azure_openai_cli_overrides(args: argparse.Namespace) -> None:
    """Copy any --azure-openai-* flags into the process environment, before
    anything else runs, so `LLMConfig.from_env()` (read later, inside
    `ForecastEnginePipeline`) sees them exactly as it would a real
    environment variable. A deployment that already sets these as real
    environment variables passes no flags here and this is a no-op.
    Values are resolved and copied, never logged or printed.
    """
    if args.azure_openai_endpoint:
        os.environ[AZURE_OPENAI_ENDPOINT_ENV_VAR] = _resolve_databricks_secret_template(args.azure_openai_endpoint)
    if args.azure_openai_api_key:
        os.environ[AZURE_OPENAI_API_KEY_ENV_VAR] = _resolve_databricks_secret_template(args.azure_openai_api_key)
    if args.azure_openai_deployment:
        os.environ[AZURE_OPENAI_DEPLOYMENT_NAME_ENV_VAR] = _resolve_databricks_secret_template(
            args.azure_openai_deployment
        )
