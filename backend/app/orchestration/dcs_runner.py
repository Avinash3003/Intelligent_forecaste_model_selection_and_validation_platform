"""DCS Runner — the Databricks Container Services variant of `DatabricksRunner`.

Submit/poll/retrieve, staging, error translation and result mapping are all
identical to Serverless execution — both are Jobs API runs against the same
UC Volume. The only thing that differs is *which deployed Job* is targeted
(the ACR/Docker job vs. the Serverless one) and which `ExecutionBackend` is
reported back. Both are constructor arguments `DatabricksRunner` already
takes, so this subclass is the whole difference.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import Settings
from app.orchestration.databricks_runner import DatabricksRunner
from app.orchestration.mlflow_history import MLflowHistoryStore
from app.orchestration.schemas import ExecutionBackend


class DcsRunner(DatabricksRunner):
    """Executes the pipeline as a run of the existing DCS (ACR/Docker) Job."""

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
