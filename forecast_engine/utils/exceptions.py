"""The engine's own exception types.

Every failure is one of these, so a caller can tell "the input was wrong"
from "the engine broke" without parsing pandas or OS error strings.
Messages are shown to operators, so raise them as complete sentences.
"""


class ForecastEngineError(Exception):
    """Base class for every error raised by the Forecast Engine."""


class DatasetLoadError(ForecastEngineError):
    """The file could not be read into a usable DataFrame — missing,
    unsupported format, unparseable, or empty."""


class ConfigurationError(ForecastEngineError):
    """The configuration contradicts itself or the dataset — a missing target
    column, or one column assigned two roles."""


class PreprocessingError(ForecastEngineError):
    """Raised when preprocessing cannot produce a usable dataset — e.g.
    every row was dropped because no date value could be parsed.
    """


class GroupGenerationError(ForecastEngineError):
    """No forecasting groups could be derived — an empty dataset, or key
    columns that yield no complete combinations."""


class DataQualityError(ForecastEngineError):
    """The data is readable and correctly typed but not good enough to
    forecast — too little history, or too many missing target values."""


class CuratedDatasetError(ForecastEngineError):
    """Raised when the curated dataset cannot be persisted (unwritable
    destination, unsupported output format).
    """


class ModelRegistryError(ForecastEngineError):
    """A model could not be resolved — unknown name, bad adapter path, or an
    adapter missing the common interface.

    An uninstalled library is not this: that is reported as Unavailable.
    """


class ForecastGenerationError(ForecastEngineError):
    """One (group, model) pair could not produce a forecast. Recorded as
    Failed and the run continues."""


class TrainingError(ForecastEngineError):
    """Training cannot proceed at all, e.g. the curated dataset failed its
    integrity check. One model failing on one group is not this."""


class LLMProviderError(ForecastEngineError):
    """The LLM provider could not complete — missing key, unreachable
    endpoint, malformed response.

    Always caught: an insights failure must never fail the forecast it
    was explaining.
    """


class MLflowTrackingError(ForecastEngineError):
    """Tracking, artifact logging or model registration failed.

    Always caught: a logging failure must never fail the run it recorded,
    and the forecast results stay fully available.
    """
