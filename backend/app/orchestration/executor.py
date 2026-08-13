"""Pipeline Executor (Section 6.14) — the single entry point the rest of
the backend is allowed to call to run a forecast.

Deliberately thin: every method is a one-line delegation to whichever
`PipelineRunner` was selected at construction time. The Executor holds no
execution logic of its own and inspects no run state directly — "the
Pipeline Executor must never know how the pipeline is executed" is
enforced by there simply being nowhere in this class for that knowledge to
live.

`get_pipeline_executor()` is a process-wide singleton (mirroring
`app.config.settings.get_settings()`'s own `@lru_cache` pattern): the
selected Runner owns in-memory job state, so every caller — the `/deploy`
route today, `/execution/*` routes, any future caller — must share the
exact same Runner instance or a `/results/{run_id}` request would never
see the job `/deploy` just submitted.
"""

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
    """Runs a forecasting pipeline through whichever Runner is configured."""

    def __init__(self, runner: PipelineRunner | None = None, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._runner = runner or build_runner(settings)

    def execute(self, request: PipelineExecutionRequest) -> str:
        """Submit `request` for execution. Returns immediately with a run id."""
        return self._runner.submit(request)

    def get_status(self, run_id: str) -> JobStatus:
        return self._runner.get_status(run_id)

    def get_result(self, run_id: str) -> PipelineExecutionResult:
        return self._runner.get_result(run_id)

    def list_runs(self) -> list[RunListing]:
        """Every run submitted to the active Runner, most recent first."""
        return self._runner.list_runs()

    def get_run(self, run_id: str) -> RunListing | None:
        """One run's listing including its stage trail, or None if unknown."""
        return self._runner.get_run(run_id)

    def cancel(
        self,
        run_id: str,
        cancelled_by_user_id: str | None = None,
        cancelled_by_display_name: str | None = None,
    ) -> CancellationOutcome:
        return self._runner.cancel(run_id, cancelled_by_user_id, cancelled_by_display_name)


def build_runner(settings: Settings) -> PipelineRunner:
    """Select the Runner `settings.execution_mode` names.

    Raises:
        RunnerConfigurationError: for any value other than the ones this
            platform supports — an unambiguous failure beats silently
            defaulting to a backend the deployment did not ask for.
    """
    try:
        mode = ExecutionBackend(settings.execution_mode)
    except ValueError as exc:
        supported = ", ".join(backend.value for backend in ExecutionBackend)
        raise RunnerConfigurationError(
            f"Unknown execution_mode '{settings.execution_mode}'. Supported values are: {supported}."
        ) from exc

    # Every branch is explicit — an `else: LocalRunner()` fallthrough would
    # silently run a future/misconfigured cloud mode locally instead of
    # failing, which is exactly what the ValueError branch above exists to
    # prevent.
    if mode is ExecutionBackend.DATABRICKS:
        return DatabricksRunner(settings)
    if mode is ExecutionBackend.DATABRICKS_DCS:
        return DcsRunner(settings)
    return LocalRunner(settings)


@lru_cache
def get_pipeline_executor() -> PipelineExecutor:
    """The process-wide Pipeline Executor. See module docstring for why
    this must be a singleton rather than constructed per call.
    """
    return PipelineExecutor()
