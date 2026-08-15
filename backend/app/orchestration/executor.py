"""The single entry point the rest of the backend uses to run a forecast."""

from __future__ import annotations

from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.orchestration.databricks_runner import DatabricksRunner
from app.orchestration.dcs_runner import DcsRunner
from app.orchestration.exceptions import RunnerConfigurationError
from app.orchestration.local_runner import LocalRunner
from app.orchestration.runner_base import PipelineRunner
from app.orchestration.schemas import (
    CancellationOutcome,
    ExecutionBackend,
    JobStatus,
    PipelineExecutionRequest,
    PipelineExecutionResult,
    RunListing,
)


class PipelineExecutor:
    """Delegates every call to the configured Runner.

    Holds no execution logic of its own, so it never needs to know whether a
    run is executing locally or on Databricks.
    """

    def __init__(self, runner: PipelineRunner | None = None, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._runner = runner or build_runner(settings)

    def execute(self, request: PipelineExecutionRequest) -> str:
        """Submit a run and return its id immediately."""
        return self._runner.submit(request)

    def get_status(self, run_id: str) -> JobStatus:
        return self._runner.get_status(run_id)

    def get_result(self, run_id: str) -> PipelineExecutionResult:
        return self._runner.get_result(run_id)

    def list_runs(self) -> list[RunListing]:
        """Every run on the active Runner, newest first."""
        return self._runner.list_runs()

    def get_run(self, run_id: str) -> RunListing | None:
        """One run including its stage trail, or None if unknown."""
        return self._runner.get_run(run_id)

    def cancel(
        self,
        run_id: str,
        cancelled_by_user_id: str | None = None,
        cancelled_by_display_name: str | None = None,
    ) -> CancellationOutcome:
        return self._runner.cancel(run_id, cancelled_by_user_id, cancelled_by_display_name)


def build_runner(settings: Settings) -> PipelineRunner:
    """Pick the Runner named by EXECUTION_MODE.

    Raises RunnerConfigurationError on an unknown mode — failing loudly beats
    silently running on a backend nobody asked for.
    """
    try:
        mode = ExecutionBackend(settings.execution_mode)
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in ExecutionBackend)
        raise RunnerConfigurationError(
            f"Unknown execution_mode '{settings.execution_mode}'. Supported values are: {supported}."
        ) from exc

    if mode is ExecutionBackend.DATABRICKS:
        return DatabricksRunner(settings)
    if mode is ExecutionBackend.DATABRICKS_DCS:
        return DcsRunner(settings)
    return LocalRunner(settings)


@lru_cache
def get_pipeline_executor() -> PipelineExecutor:
    """Process-wide singleton.

    The Runner holds in-memory job state, so every caller must share one
    instance or a status poll would not find the run that /deploy just
    submitted.
    """
    return PipelineExecutor()
