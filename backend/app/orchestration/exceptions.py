"""Orchestration errors, so API routes catch one predictable type."""


class ExecutionError(Exception):
    """Base class for every orchestration failure."""


class UnknownRunError(ExecutionError):
    """The run_id belongs to no job this Runner submitted."""


class RunNotReadyError(ExecutionError):
    """A result was requested before the run finished. Poll status first."""


class RunnerConfigurationError(ExecutionError):
    """The Runner cannot start at all — bad execution_mode, or a missing engine interpreter."""
