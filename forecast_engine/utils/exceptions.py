"""Domain exceptions for the Forecast Engine.

Every failure the engine can produce is expressed as one of these types so
callers (the local CLI today, a Databricks task in a later phase) can
distinguish "the input was wrong" from "the engine broke" without parsing
pandas or OS error strings. Messages are written to be surfaced directly to
an operator, so raise them with complete sentences.
"""


class ForecastEngineError(Exception):
    """Base class for every error raised by the Forecast Engine."""


class DatasetLoadError(ForecastEngineError):
    """Raised when a dataset file cannot be read into a usable DataFrame.

    Covers missing files, unsupported formats, unparseable content and
    files that contain no rows/columns.
    """


class ConfigurationError(ForecastEngineError):
    """Raised when a ForecastConfiguration is internally inconsistent or
    does not match the dataset it is applied to — e.g. the declared target
    column is absent, or a column has been assigned two metadata roles.
    """


class PreprocessingError(ForecastEngineError):
    """Raised when preprocessing cannot produce a usable dataset — e.g.
    every row was dropped because no date value could be parsed.
    """


class GroupGenerationError(ForecastEngineError):
    """Raised when no forecasting groups could be derived from the
    prepared dataset (an empty dataset, or key columns that yield no
    complete key combinations).
    """


class DataQualityError(ForecastEngineError):
    """Raised when a dataset fails a quality gate that makes forecasting
    impossible — too few historical observations, or so many missing target
    values that dropping the affected rows would destroy the history.

    Distinct from PreprocessingError: the data is readable and typed
    correctly, it simply isn't good enough to forecast.
    """


class CuratedDatasetError(ForecastEngineError):
    """Raised when the curated dataset cannot be persisted (unwritable
    destination, unsupported output format).
    """


class ModelRegistryError(ForecastEngineError):
    """Raised when a model cannot be resolved from the registry — an
    unknown model name, a bad adapter import path, or an adapter that does
    not implement the common interface.

    All three are deployment/configuration errors. A merely *uninstalled*
    ML library is not one of them: that is an expected condition reported
    as an Unavailable model, never raised.
    """


class ForecastGenerationError(ForecastEngineError):
    """Raised when a model cannot produce a complete forward forecast.

    Scoped to one (group, model) pair: the evaluation stage records the
    pair as Failed and continues with the rest, so this never ends a run.
    """


class TrainingError(ForecastEngineError):
    """Raised when training cannot proceed at all — for example when the
    curated dataset fails its pre-training integrity check.

    A single model failing on a single group is NOT this: those are
    captured per model so the rest of the run continues.
    """


class LLMProviderError(ForecastEngineError):
    """Raised when an LLM provider cannot produce a completion — a missing
    API key, an unreachable endpoint, or a malformed response.

    Always caught by the LLM Insight Engine, never left to propagate: a
    business-insights failure must never fail the forecasting pipeline that
    produced the results it was explaining.
    """


class MLflowTrackingError(ForecastEngineError):
    """Raised when MLflow experiment tracking, artifact logging, or model
    registration cannot complete — an unreachable tracking server, an
    invalid tracking URI, or a registry write failure.

    Always caught by the tracking pipeline, never left to propagate: a
    logging failure must never fail the forecasting run it was recording.
    The forecast results remain fully available even when this fires.
    """
