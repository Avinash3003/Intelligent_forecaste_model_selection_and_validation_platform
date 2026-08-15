"""The Container Services variant of DatabricksRunner.

Everything (staging, submit, poll, result mapping) is identical to Serverless
execution — only the target job and the reported backend differ, and both are
constructor arguments, so this subclass is the whole difference.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import Settings
from app.orchestration.databricks_runner import DatabricksRunner
from app.orchestration.mlflow_history import MLflowHistoryStore
from app.orchestration.schemas import ExecutionBackend


class DcsRunner(DatabricksRunner):
    """Runs the pipeline as a Databricks Container Services job."""

    def __init__(
        self,
        settings: Settings,
        history: MLflowHistoryStore | None = None,
        workspace_client: Any | None = None,
    ) -> None:
        super().__init__(
            settings,
            history=history,
            workspace_client=workspace_client,
            execution_backend=ExecutionBackend.DATABRICKS_DCS,
            job_name=settings.databricks_dcs_job_name,
            job_id=settings.databricks_dcs_job_id,
        )
