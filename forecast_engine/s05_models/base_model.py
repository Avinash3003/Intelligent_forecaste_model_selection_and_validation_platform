"""Base forecasting model interface.

Every model family the platform supports — statistical, gradient-boosted,
deep — is reached through this one interface. The trainer therefore never
knows which library it is driving, which is what lets a new model be
onboarded by registering an adapter rather than by editing the training
loop (Section 6.4, "pluggable model registry").

Two responsibilities live in the adapter, deliberately:

  * Library isolation. Each adapter imports its backing library lazily
    inside `initialize()`, so a deployment that never selects TFT is not
    forced to install PyTorch, and a missing library degrades to a skipped
    model rather than a crashed pipeline.
  * Input adaptation. Every library wants a different shape — Prophet
    wants ds/y columns, ARIMA a univariate series, trees a numeric design
    matrix. Converting the common ForecastSeries into that shape belongs
    to the adapter that needs it, not to the shared pipeline. This is
    input plumbing only; genuine feature engineering (lags, rolling
    windows, encodings — Section 6.3) is a separate later phase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd

from forecast_engine.config.model_config import ModelSpec
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries


class TrainingStatus(str, Enum):
    """Outcome of training one model on one forecasting group.

    SKIPPED and UNAVAILABLE are distinct from FAILED on purpose: neither
    indicates a defect. A group too short for a deep model, or a library a
    deployment chose not to install, are expected conditions that later
    stages should treat differently from a genuine training failure.
    """

    TRAINED = "Trained"
    FAILED = "Failed"
    SKIPPED = "Skipped"
    UNAVAILABLE = "Unavailable"


@dataclass
class ForecastOutput:
    """Point forecasts and, where the model supports them, intervals.

    Confidence intervals are optional by design: statistical models
    (ARIMA, Prophet) produce them natively, whereas the gradient-boosted
    regressors do not. Rather than fabricate intervals for the trees —
    which would misrepresent their uncertainty — the fields stay None and
    downstream consumers check before using them.
    """

    values: list[float] = field(default_factory=list)
    lower: list[float] | None = None
    upper: list[float] | None = None

    # Whether both interval bounds were produced
    @property
    def has_intervals(self) -> bool:
        return self.lower is not None and self.upper is not None

    # Serializable representation of the forecast output
    def to_dict(self) -> dict[str, Any]:
        return {
            "values": _jsonable(self.values),
            "lower": _jsonable(self.lower) if self.lower is not None else None,
            "upper": _jsonable(self.upper) if self.upper is not None else None,
        }


@dataclass
class TrainedModel:
    """One trained model, its provenance and its outcome.

    This is the unit of output for the whole phase: the next phase consumes
    a collection of these. The fitted estimator itself is held in `model`
    and excluded from `to_dict()`, which carries only the serializable
    training record.
    """

    group_id: str
    model_name: str
    status: TrainingStatus
    model: Any | None = None
    key_values: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    trained_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # The fitted `BaseForecastingModel` wrapper itself (not just its raw
    # `.model` estimator), kept so a later stage can call `.predict()`
    # directly instead of reconstructing and refitting an identical model.
    # Excluded from `to_dict()` like `model` — it is a live in-process
    # object, not part of the run's serializable record. `predict()` reads
    # only the state `train()` already wrote (see `SupervisedTreeModel.
    # predict`'s local `history` copy) and no stage mutates a fitted
    # wrapper, so calling it more than once is safe.
    fitted_model: Any = None

    # Whether this record represents a successfully trained model
    @property
    def is_trained(self) -> bool:
        return self.status is TrainingStatus.TRAINED

    # Serializable training record, excluding the estimator object
    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "model_name": self.model_name,
            "status": self.status.value,
            "key_values": self.key_values,
            "params": _jsonable(self.params),
            "metadata": _jsonable(self.metadata),
            "error": self.error,
            "trained_at": self.trained_at.isoformat(timespec="seconds"),
        }


# Coerce numpy/pandas scalars and containers into plain JSON types
def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class BaseForecastingModel(ABC):
    """Common interface implemented by every forecasting model adapter."""

    # Merge spec defaults with any override params and reset fitted state
    def __init__(self, spec: ModelSpec, params: dict[str, Any] | None = None) -> None:
        self.spec = spec
        self.params: dict[str, Any] = {**spec.default_params, **(params or {})}
        self._model: Any | None = None

    # This model's registered name
    @property
    def name(self) -> str:
        return self.spec.name

    # Whether the model has been fitted
    @property
    def is_trained(self) -> bool:
        return self._model is not None

    # The fitted estimator, or None before training
    @property
    def model(self) -> Any | None:
        return self._model

    # Whether this model's backing library is importable
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool: ...

    # Construct the underlying estimator from self.params
    @abstractmethod
    def initialize(self) -> None: ...

    # Fit the model on one forecasting group's series
    @abstractmethod
    def train(self, series: ForecastSeries) -> dict[str, Any]: ...

    # Forecast horizon periods beyond the fitted training window
    @abstractmethod
    def predict(self, horizon: int, future_frame: pd.DataFrame | None = None) -> ForecastOutput: ...

    # Whether a hyperparameter search is meaningful for this model
    def supports_tuning(self) -> bool:
        return bool(self.spec.search_space)

    # Short, serializable description of this model instance
    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "params": _jsonable(self.params), "trained": self.is_trained}

    # Guard prediction against an unfitted model
    def _require_trained(self) -> None:
        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' must be trained before it can forecast.")

    # ------------------------------------------------------------------
    # Shared input helpers
    # ------------------------------------------------------------------

    # Return the target as a date-indexed, numeric series
    def _target_series(self, series: ForecastSeries) -> pd.Series:
        frame = series.frame
        values = pd.Series(
            frame[series.target_column].to_numpy(),
            index=pd.DatetimeIndex(frame[series.date_column]),
            name=series.target_column,
        )
        return values.astype(float)

    # Declared feature columns that are numeric enough to train on
    def _usable_feature_columns(self, series: ForecastSeries) -> list[str]:
        # Non-numeric regressors are dropped rather than encoded: encoding is
        # feature engineering that belongs to a later phase.
        if not self.spec.supports_features:
            return []

        usable = []
        for column in series.feature_columns:
            if column in series.frame.columns and pd.api.types.is_numeric_dtype(series.frame[column]):
                usable.append(column)
        return usable


class SupervisedTreeModel(BaseForecastingModel):
    """Shared behaviour for gradient-boosted tree adapters.

    XGBoost and LightGBM differ only in which estimator class they build; the
    way a time series is framed as a supervised problem is identical, so it
    lives here rather than being duplicated in both adapters.

    Feature engineering (Section 6.3). A tree splits on feature values it saw
    in training, so it cannot extrapolate an ordinal `time_index` past the last
    training row — every future step lands in the same terminal leaf and the
    forecast comes out dead flat. That is not a tuning problem: it is a
    consequence of framing the series with position as its only predictor. The
    lag, rolling and calendar features below give the tree something it *can*
    split on out of sample, which is what lets it reproduce level, trend and
    seasonality instead of a horizontal line.

    Every generated column is configuration-driven through the model spec's
    params (`lags`, `rolling_windows`, `calendar_features`), so the feature set
    is changed by configuration rather than by editing this class.
    """

    # Defaults chosen for the platform's monthly grain: the previous month
    # carries level, lag 12 carries annual seasonality, and a 3-month mean
    # smooths noise without erasing the trend.
    DEFAULT_LAGS = (1, 2, 3, 12)
    DEFAULT_ROLLING_WINDOWS = (3, 6)

    # A tree can only predict values bracketed by what it saw in training, so
    # on a trending series it saturates at the highest training level and the
    # forecast flattens. Fitting the first difference sidesteps that: the
    # change month-to-month is roughly stationary even when the level is not,
    # and the level is rebuilt by accumulating predicted changes. Set
    # `target_transform: "none"` in the model params to fit levels directly.
    DEFAULT_TARGET_TRANSFORM = "difference"

    # Populated by build_design_matrix from the series' own length; declared
    # here so _feature_row has a schema even if called before a fit.
    _lags: tuple[int, ...] = DEFAULT_LAGS
    _windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    # At least this many usable rows must survive lag construction, otherwise
    # there is nothing left to fit.
    MIN_TRAINING_ROWS = 8

    def _configured_lags(self, observations: int | None = None) -> tuple[int, ...]:
        # `.get(key, default)`, not `.get(key) or default`: an *absent* key
        # (every caller before per-feature selection existed) means "use
        # the default", but an explicit empty list (a user who deselected
        # every lag) must stay empty, not silently fall back to it.
        return self._fit_to_history(tuple(self.params.get("lags", self.DEFAULT_LAGS)), observations)

    def _configured_windows(self, observations: int | None = None) -> tuple[int, ...]:
        return self._fit_to_history(
            tuple(self.params.get("rolling_windows", self.DEFAULT_ROLLING_WINDOWS)), observations
        )

    def _fit_to_history(self, spans: tuple[int, ...], observations: int | None) -> tuple[int, ...]:
        """Drop spans the series is too short to support.

        A lag of 12 needs twelve prior observations, so every row before that
        is incomplete and gets dropped from training. On a short slice — a
        backtest window starting at the configured minimum of 12 — that leaves
        nothing to fit, and the model silently produces no backtest at all.
        Narrowing the spans to what the history can actually support keeps the
        feature set honest for short series instead of emptying the matrix.

        An empty `spans` is left alone regardless of `observations`: that is
        a deliberate "none of these" (a user who selected zero lags/windows,
        Priority C), not a series too short to support what was asked —
        the floor below exists only for the latter.
        """
        if observations is None or not spans:
            return spans
        usable = tuple(s for s in spans if observations - s >= self.MIN_TRAINING_ROWS)
        # Lag 1 is always affordable and is what carries the level, so it is
        # kept as the floor rather than returning no features at all.
        return usable or (1,)

    # Names of the calendar columns to generate. Accepts the historical
    # `calendar_features: bool` (True = both, False/absent-with-explicit-
    # False = neither) as well as an explicit list of names (Priority C's
    # per-run derived-feature selection) — a caller that has never heard of
    # per-feature selection keeps working unchanged.
    _CALENDAR_FEATURE_NAMES = ("month", "quarter")

    def _calendar_features(self) -> tuple[str, ...]:
        configured = self.params.get("calendar_features", True)
        if isinstance(configured, bool):
            return self._CALENDAR_FEATURE_NAMES if configured else ()
        return tuple(name for name in configured if name in self._CALENDAR_FEATURE_NAMES)

    def _differenced(self) -> bool:
        return (self.params.get("target_transform") or self.DEFAULT_TARGET_TRANSFORM) == "difference"

    # `self.params` also carries these feature-engineering knobs (read by
    # `_configured_lags`/`_configured_windows`/`_calendar_features` above),
    # which are never valid keyword arguments for the underlying sklearn-style
    # estimator — `build_estimator()` must construct it from `self.params`
    # with these removed, not `self.params` verbatim.
    _FEATURE_ENGINEERING_PARAM_KEYS = frozenset({"lags", "rolling_windows", "calendar_features", "target_transform"})

    def _estimator_params(self) -> dict[str, Any]:
        return {k: v for k, v in self.params.items() if k not in self._FEATURE_ENGINEERING_PARAM_KEYS}

    def _feature_row(
        self, history: list[float], position: int, timestamp: pd.Timestamp | None
    ) -> dict[str, float]:
        """Build one row of features from the values observed *before* it.

        `history` holds every target value up to (not including) this row, so
        the same function serves training and forecasting: during training it
        is fed actuals, during prediction it is fed actuals followed by the
        model's own predictions. Using one builder for both is what guarantees
        the forecast-time matrix matches the one the model was fitted on.
        """
        row: dict[str, float] = {"time_index": float(position)}

        for lag in self._lags:
            row[f"lag_{lag}"] = float(history[-lag]) if len(history) >= lag else float("nan")

        for window in self._windows:
            if len(history) >= window:
                recent = history[-window:]
                row[f"rolling_mean_{window}"] = float(sum(recent) / window)
            else:
                row[f"rolling_mean_{window}"] = float("nan")

        # Month and quarter let the tree separate seasonal positions that an
        # ordinal index cannot express once it runs past the training range.
        if timestamp is not None:
            calendar_features = self._calendar_features()
            if "month" in calendar_features:
                row["month"] = float(timestamp.month)
            if "quarter" in calendar_features:
                row["quarter"] = float(timestamp.quarter)

        return row

    # Frame series as (features, target) for a supervised learner
    def build_design_matrix(self, series: ForecastSeries) -> tuple[pd.DataFrame, pd.Series]:
        frame = series.frame
        values = frame[series.target_column].astype(float).tolist()
        dates = (
            pd.to_datetime(frame[series.date_column]).tolist()
            if series.date_column in frame.columns
            else [None] * len(frame)
        )

        # Spans are chosen once per fit from this series' own length, then
        # reused unchanged at predict time so both matrices share a schema.
        self._lags = self._configured_lags(len(frame))
        self._windows = self._configured_windows(len(frame))

        rows = [self._feature_row(values[:index], index, dates[index]) for index in range(len(frame))]
        features = pd.DataFrame(rows, index=frame.index)

        for column in self._usable_feature_columns(series):
            features[column] = frame[column].to_numpy()

        levels = frame[series.target_column].astype(float)
        # Row i predicts the change from i-1 to i; the first row has no prior
        # value and is dropped with the other incomplete rows below.
        target = levels.diff() if self._differenced() else levels

        # The earliest rows have no history to lag against. Dropping them is
        # preferable to imputing a value the series never had, which would
        # teach the model a relationship that does not exist.
        complete = features.notna().all(axis=1) & target.notna()
        if complete.any():
            features, target = features[complete], target[complete]

        return features, target

    # Construct the concrete regressor for this library
    @abstractmethod
    def build_estimator(self) -> Any: ...

    # Build the underlying estimator instance
    def initialize(self) -> None:
        self._estimator = self.build_estimator()

    # Fit the estimator on the series' design matrix
    def train(self, series: ForecastSeries) -> dict[str, Any]:
        if getattr(self, "_estimator", None) is None:
            self.initialize()

        features, target = self.build_design_matrix(series)
        self._estimator.fit(features, target)
        self._model = self._estimator

        frame = series.frame
        self._feature_names = list(features.columns)
        self._training_length = int(len(frame))
        # Kept whole (not just the last row) because recursive forecasting
        # needs to look back as far as the longest configured lag.
        self._target_history = frame[series.target_column].astype(float).tolist()
        self._last_timestamp = (
            pd.to_datetime(frame[series.date_column]).iloc[-1]
            if series.date_column in frame.columns and len(frame)
            else None
        )
        self._observed_step = self._infer_step(frame, series)
        self._last_feature_row = features.iloc[-1].to_dict() if not features.empty else {}

        return {
            "observations": int(len(target)),
            "feature_count": int(features.shape[1]),
            "features_used": self._feature_names,
            "lags": list(self._lags),
            "rolling_windows": list(self._windows),
        }

    def _infer_step(self, frame: pd.DataFrame, series: ForecastSeries) -> pd.Timedelta:
        """Spacing between observations, so forecast timestamps continue the
        series' own grain rather than assuming monthly."""
        if series.date_column not in frame.columns or len(frame) < 2:
            return pd.Timedelta(days=30)
        gaps = pd.to_datetime(frame[series.date_column]).diff().dropna()
        median = gaps.median()
        return median if pd.notna(median) and median > pd.Timedelta(0) else pd.Timedelta(days=30)

    # Predict horizon steps recursively, feeding each prediction back as history
    def predict(self, horizon: int, future_frame: pd.DataFrame | None = None) -> ForecastOutput:
        """Forecast one step at a time, appending each prediction to history.

        Multi-step forecasting with lag features has to be recursive: step 2's
        `lag_1` is step 1's prediction. The previous implementation instead
        froze every non-index feature at its last observed value, which is the
        direct cause of the flat forecasts that forward validation was
        eliminating these models for.
        """
        self._require_trained()

        history = list(self._target_history)
        timestamp = self._last_timestamp
        predictions: list[float] = []

        for step in range(horizon):
            timestamp = timestamp + self._observed_step if timestamp is not None else None
            row = self._feature_row(history, self._training_length + step, timestamp)

            # Declared regressors come from future_frame when the caller has
            # future values; otherwise the last observed value is carried
            # forward, which is explicit rather than silently zero.
            for column in self._feature_names:
                if column in row:
                    continue
                if future_frame is not None and column in future_frame.columns and step < len(future_frame):
                    row[column] = float(future_frame[column].iloc[step])
                else:
                    row[column] = float(self._last_feature_row.get(column, 0.0))

            design = pd.DataFrame([row])[self._feature_names].fillna(
                pd.Series(self._last_feature_row)
            )
            raw = float(self._model.predict(design)[0])
            # `raw` is a predicted change when the target was differenced, so
            # the level is rebuilt by adding it to the previous level.
            value = history[-1] + raw if self._differenced() and history else raw
            predictions.append(value)
            history.append(value)

        return ForecastOutput(values=predictions)
