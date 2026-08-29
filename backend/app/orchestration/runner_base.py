"""The interface every execution backend (local, Databricks) implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.orchestration.schemas import (
    CancellationOutcome,
    JobStatus,
    PipelineExecutionRequest,
    PipelineExecutionResult,
    RunListing,
)


class PipelineRunner(ABC):
    """Submit a run, poll its status, fetch its result.

    The shape is submit/poll/retrieve rather than one blocking run() because
    a Databricks job is asynchronous; giving local execution the same shape
    keeps the two backends interchangeable.
    """

    @abstractmethod
    def submit(self, request: PipelineExecutionRequest) -> str:
        """Start the run and return its id immediately, without waiting."""

    @abstractmethod
    def get_status(self, run_id: str) -> JobStatus:
        """Current status. Raises UnknownRunError if the id is not ours."""

    @abstractmethod
    def get_result(self, run_id: str) -> PipelineExecutionResult:
        """Result of a finished run. Raises RunNotReadyError if still going."""

    @abstractmethod
    def list_runs(self) -> list[RunListing]:
        """Every run this backend knows about, newest first, without stage detail."""

    @abstractmethod
    def get_run(self, run_id: str) -> RunListing | None:
        """One run including its stage trail, or None if unknown."""

    def prewarm(self) -> None:
        """Start whatever slow first read this backend has, off the request path.

        Optional, and deliberately not abstract: a backend with nothing
        slow to warm does nothing here. Must return immediately — it is
        called during application startup, not from a request.
        """

    @abstractmethod
    def cancel(
        self,
        run_id: str,
        cancelled_by_user_id: str | None = None,
        cancelled_by_display_name: str | None = None,
    ) -> CancellationOutcome:
        """Stop a pending/running job and delete whatever it already wrote.

        Idempotent: cancelling an already-finished run returns cancelled=False
        rather than raising. cleanup_errors names any storage location that
        could not be removed.
        """
