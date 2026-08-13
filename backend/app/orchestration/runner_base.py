"""The common Runner interface (Section 6.14, "Runner Architecture").

Every execution backend — Local today, Databricks later — implements this
exact interface. The Pipeline Executor is written against this ABC only;
it is what lets `PipelineExecutor` "never know how the pipeline is
executed" while still being able to run it.

The interface is submit/poll/retrieve rather than a single blocking `run()`
call, deliberately: a Databricks Job is inherently asynchronous (submit,
then poll for completion), and giving Local execution the identical shape
— submit returns immediately, status is polled, the result is fetched once
terminal — is what makes the two backends genuinely interchangeable rather
than superficially similar.
"""

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
    """Common interface every execution backend must implement."""

    @abstractmethod
    def submit(self, request: PipelineExecutionRequest) -> str:
        """Begin executing `request` and return immediately.

        Returns:
            The run id future calls identify this execution by (either the
            id `request.run_id` already carried, or one generated here).

        Never blocks until completion — the caller polls `get_status()` /
        calls `get_result()` once the status is terminal.
        """

    @abstractmethod
    def get_status(self, run_id: str) -> JobStatus:
        """Current status of a previously submitted run.

        Raises:
            UnknownRunError: if `run_id` was never submitted to this Runner.
        """

    @abstractmethod
    def get_result(self, run_id: str) -> PipelineExecutionResult:
        """The standardized result of a completed run.

        Raises:
            UnknownRunError: if `run_id` was never submitted.
            RunNotReadyError: if the run has not yet reached a terminal
                status.
        """

    @abstractmethod
    def list_runs(self) -> list[RunListing]:
        """Every run this Runner has been given, most recent first.

        Backs the run-history view. Returns listings rather than full
        results so rendering a history page never costs one full result
        deserialization per run.
        """

    @abstractmethod
    def get_run(self, run_id: str) -> RunListing | None:
        """One run's listing, including its stage trail, or None if unknown.

        The detail counterpart to `list_runs()`: the list view stays cheap
        by omitting per-run stage detail, and this loads it for the single
        run actually being inspected.
        """

    @abstractmethod
    def cancel(
        self,
        run_id: str,
        cancelled_by_user_id: str | None = None,
        cancelled_by_display_name: str | None = None,
    ) -> CancellationOutcome:
        """Attempt to cancel a pending or running job.

        On acceptance, this also reconciles everything the run may have
        already written: every run-scoped storage location is deleted
        (strictly by `run_id`, never touching another run's data), and any
        MLflow Parent Run already opened for it is marked terminated rather
        than left `RUNNING` forever. `cancelled_by_*` is recorded alongside
        the run's own `started_by_*` so a run history entry can show both.

        Idempotent: calling this again on an already-cancelled run returns
        `cancelled=False` and performs no further action, rather than
        erroring or repeating the cleanup.

        Returns:
            A `CancellationOutcome`. `cancelled=False` if the job had
            already reached a terminal status and there was nothing to
            cancel — not an error. `cleanup_errors` names exactly which
            storage location(s) could not be cleaned up, if any; an empty
            list means every one succeeded (or was already clean).

        Raises:
            UnknownRunError: if `run_id` was never submitted.
        """
