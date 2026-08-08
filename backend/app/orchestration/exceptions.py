"""Orchestration-layer exceptions.

Every failure mode a Runner or the Pipeline Executor can hit is expressed
as one of these, so an API route catches a single predictable type — never
a raw `subprocess.SubprocessError`, `FileNotFoundError`, or SDK exception —
and translates it into a clean HTTP response.
"""


class ExecutionError(Exception):
    """Base class for every error the orchestration layer raises."""


class UnknownRunError(ExecutionError):
    """Raised when a `run_id` does not correspond to any job the selected
    Runner has ever submitted — an invalid id, or one from a previous
    process (Local execution's job registry is in-memory only).
    """


class RunNotReadyError(ExecutionError):
    """Raised when a result is requested for a run that has not yet
    reached a terminal status (COMPLETED/FAILED/CANCELLED). The caller
    should poll status first.
    """


class RunnerConfigurationError(ExecutionError):
    """Raised when the selected Runner cannot even attempt execution — an
    unknown `execution_mode`, or a LocalRunner whose configured
    forecast_engine interpreter does not exist.
    """


class DatabricksNotImplementedError(ExecutionError):
    """Raised by every placeholder integration point in `DatabricksRunner`.

    Section 6.14 explicitly scopes this phase to architecture only — no
    real Azure Storage or Databricks Jobs API call is made. Each stub
    raises this with a message naming exactly what a later phase must
    implement.
    """
