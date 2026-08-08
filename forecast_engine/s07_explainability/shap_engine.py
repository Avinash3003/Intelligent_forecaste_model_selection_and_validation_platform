"""SHAP / native-importance explainability engine (Section 6.10).

For models supporting SHAP directly (the tree-based adapters, when the
optional `shap` package is installed) this computes real SHAP values, both
global (mean absolute contribution per feature) and local (per-observation
attributions). Every other model — ARIMA, Prophet, TFT, or a tree model when
`shap` is not installed — is scored through a model-agnostic
permutation-importance path built on the shared `predict()` interface, so no
surviving model is ever skipped.

Four structured signals are produced either way, sharing the same
combination logic regardless of which path produced the underlying
importance vectors:

  * window stability — how much the importance ranking moves across
    independent chronological segments of history.
  * horizon stability — how much it moves across different forecast-horizon
    lengths (a short holdout vs. the full configured holdout).
  * plausibility — whether importance is concentrated on features that
    actually correlate with the target.
  * dominant-feature consistency — how often the single most important
    feature stays the same across every perturbation above.

A model with no permutable/explainable feature at all (a purely univariate
statistical fit, most commonly ARIMA) has nothing to be inconsistent about,
so it receives the configured neutral fallback instead of being penalized
for a capability it was never given data to exercise.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import replace as dataclass_replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from forecast_engine.config.explainability_config import ExplainabilityConfig
from forecast_engine.s06_evaluation.metrics import MetricsCalculator
from forecast_engine.s07_explainability.explainability_report import ExplainabilityResult, StabilityMetrics
from forecast_engine.s05_models.base_model import BaseForecastingModel, SupervisedTreeModel, TrainedModel
from forecast_engine.s05_models.model_registry import ModelRegistry
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries

_SHAP_AVAILABLE = importlib.util.find_spec("shap") is not None

# Importance vectors are dicts of feature_name -> non-negative weight,
# normalized to sum to 1.0 so they are comparable across models and paths.
ImportanceVector = dict[str, float]


class ShapExplainabilityEngine:
    """Computes the full explainability record for one (group, model) pair."""

    # Wire up the model registry and explainability configuration
    def __init__(self, registry: ModelRegistry, config: ExplainabilityConfig | None = None) -> None:
        self._registry = registry
        self._config = config or ExplainabilityConfig()
        self._calculator = MetricsCalculator()

    # Score one (group, model) pair; never raises, degrades to fallback instead
    def compute(self, trained: TrainedModel, series: ForecastSeries) -> ExplainabilityResult:
        base = ExplainabilityResult(group_id=trained.group_id, model_name=trained.model_name, key_values=trained.key_values)

        if not self._config.enabled:
            return self._fallback(base, "SHAP/importance generation disabled by configuration.")

        try:
            spec = self._registry.config.find(trained.model_name)
            if spec is None:
                return self._fallback(base, "Model is no longer registered.")

            adapter_class = self._registry.load_adapter(spec)

            if issubclass(adapter_class, SupervisedTreeModel):
                return self._score_tree(base, trained, spec, series)

            return self._score_permutation(base, trained, spec, series)
        except Exception as exc:  # noqa: BLE001 - explainability must never eliminate a model
            return self._fallback(base, f"Importance computation failed: {type(exc).__name__}: {exc}")

    # Build the neutral fallback result with a recorded skip reason
    def _fallback(self, base: ExplainabilityResult, reason: str) -> ExplainabilityResult:
        base.method = "univariate_fallback"
        base.skipped_reason = reason
        base.stability = StabilityMetrics(
            window_stability=1.0,
            horizon_stability=1.0,
            plausibility=1.0,
            dominant_feature=None,
            dominant_feature_consistency=1.0,
            significant_changes=[],
            composite_score=self._config.univariate_fallback_score,
        )
        return base

    # ------------------------------------------------------------------
    # Tree models — SHAP when available, native feature_importances_ otherwise
    # ------------------------------------------------------------------

    # Score a supervised tree model via SHAP or native importances
    def _score_tree(
        self, base: ExplainabilityResult, trained: TrainedModel, spec, series: ForecastSeries
    ) -> ExplainabilityResult:
        method = "shap" if _SHAP_AVAILABLE else "native_importance"

        # Point estimate reuses the already-fitted Phase 6 model — no refit
        # needed just to read its importances.
        adapter = self._registry.create(spec, trained.params)
        features, _ = adapter.build_design_matrix(series) if isinstance(adapter, SupervisedTreeModel) else (None, None)
        if features is None or features.empty or trained.model is None:
            return self._fallback(base, "No design matrix could be reconstructed for this model.")

        local_importance: list[dict[str, Any]] = []
        if method == "shap":
            point_importance, raw_values = _shap_importance_and_values(trained.model, features)
            local_importance = _local_importance_records(raw_values, features, series, self._config.max_local_importance_rows)
        else:
            point_importance = _native_importance(trained.model, features)

        extractor: Callable[[Any, pd.DataFrame], ImportanceVector] = (
            _shap_importance if method == "shap" else _native_importance
        )

        if not point_importance:
            return self._fallback(base, "Model exposes no usable feature importances.")

        bootstrap_vectors = self._bootstrap_tree_importance(spec, trained.params, series, extractor)
        window_vectors = self._windowed_tree_importance(spec, trained.params, series, extractor)
        horizon_vectors = self._horizon_tree_importance(spec, trained.params, series, extractor, point_importance)

        return self._assemble(
            base, method, point_importance, local_importance, method == "shap",
            series, features, bootstrap_vectors, window_vectors, horizon_vectors,
        )

    # Bootstrap-resample the training frame and re-extract importance each time
    def _bootstrap_tree_importance(
        self, spec, params: dict[str, Any], series: ForecastSeries, extractor: Callable
    ) -> list[ImportanceVector]:
        vectors: list[ImportanceVector] = []
        rng = np.random.default_rng(42)
        frame = series.frame
        n = len(frame)

        for _ in range(self._config.bootstrap_iterations):
            idx = np.sort(rng.integers(0, n, size=n))
            resampled = frame.iloc[idx].reset_index(drop=True)
            vector = self._refit_and_extract(spec, params, dataclass_replace(series, frame=resampled), extractor)
            if vector:
                vectors.append(vector)
        return vectors

    # Recompute importance on independent chronological segments of history
    def _windowed_tree_importance(
        self, spec, params: dict[str, Any], series: ForecastSeries, extractor: Callable
    ) -> list[ImportanceVector]:
        segments = _chronological_segments(
            series.frame, self._config.consistency_windows, self._config.min_observations_per_window
        )
        vectors: list[ImportanceVector] = []
        for segment in segments:
            vector = self._refit_and_extract(spec, params, dataclass_replace(series, frame=segment), extractor)
            if vector:
                vectors.append(vector)
        return vectors

    # Importance recomputed on the trailing fraction of history that each
    # configured horizon fraction represents — a distinct axis from
    # `_windowed_tree_importance`'s fixed chronological segments
    def _horizon_tree_importance(
        self,
        spec,
        params: dict[str, Any],
        series: ForecastSeries,
        extractor: Callable,
        point_importance: ImportanceVector,
    ) -> list[ImportanceVector]:
        n = len(series.frame)
        vectors: list[ImportanceVector] = []
        for fraction in self._config.horizon_fractions:
            if fraction >= 0.999:
                vectors.append(point_importance)
                continue
            cutoff = max(self._config.min_observations_per_window, int(n * fraction))
            partial = series.frame.tail(cutoff).reset_index(drop=True)
            vector = self._refit_and_extract(spec, params, dataclass_replace(series, frame=partial), extractor)
            if vector:
                vectors.append(vector)
        return vectors

    # Refit on a series variant and extract its importance vector, or None on failure
    def _refit_and_extract(
        self, spec, params: dict[str, Any], series: ForecastSeries, extractor: Callable
    ) -> ImportanceVector | None:
        try:
            adapter = self._registry.create(spec, params)
            adapter.initialize()
            adapter.train(series)
            features, _ = adapter.build_design_matrix(series)
            return extractor(adapter.model, features)
        except Exception:  # noqa: BLE001 - one bad resample/segment/horizon is simply dropped
            return None

    # ------------------------------------------------------------------
    # Every other model family — permutation importance via predict()
    # ------------------------------------------------------------------

    # Score any model family via permutation importance on a holdout window
    def _score_permutation(
        self, base: ExplainabilityResult, trained: TrainedModel, spec, series: ForecastSeries
    ) -> ExplainabilityResult:
        feature_columns = _usable_permutation_features(spec, series)
        if not feature_columns:
            return self._fallback(
                base,
                "Model has no exogenous feature columns to permute; importance is attributed "
                "entirely to its own historical structure.",
            )

        holdout_size = min(max(3, len(series.frame) // 5), 12)
        if len(series.frame) <= holdout_size + spec.min_observations:
            return self._fallback(base, "Series too short to hold out a permutation-importance window.")

        train_frame = series.frame.iloc[:-holdout_size].reset_index(drop=True)
        holdout_frame = series.frame.iloc[-holdout_size:].reset_index(drop=True)
        train_series = dataclass_replace(series, frame=train_frame)

        adapter = self._registry.create(spec, trained.params)
        adapter.initialize()
        adapter.train(train_series)

        actual = holdout_frame[series.target_column].astype(float).to_numpy()

        point_importance = self._permute(adapter, holdout_frame, actual, feature_columns, seed=42)
        if not point_importance:
            return self._fallback(base, "Permutation importance produced no measurable signal.")

        stability_vectors = [
            self._permute(adapter, holdout_frame, actual, feature_columns, seed=seed)
            for seed in range(1000, 1000 + self._config.bootstrap_iterations)
        ]
        window_vectors = self._windowed_permutation_importance(spec, trained.params, series, feature_columns)
        horizon_vectors = self._horizon_permutation_importance(
            adapter, holdout_frame, actual, feature_columns, point_importance
        )

        # Permutation importance has no per-instance attribution mechanism —
        # only aggregate feature-level degradation — so local importance is
        # left empty rather than fabricated, matching "SHAP Values (where
        # applicable)".
        return self._assemble(
            base, "permutation", point_importance, [], False,
            series, None, [v for v in stability_vectors if v], window_vectors, horizon_vectors,
        )

    # Recompute permutation importance on independent chronological segments
    def _windowed_permutation_importance(
        self, spec, params: dict[str, Any], series: ForecastSeries, feature_columns: tuple[str, ...]
    ) -> list[ImportanceVector]:
        segments = _chronological_segments(
            series.frame, self._config.consistency_windows, self._config.min_observations_per_window
        )
        vectors: list[ImportanceVector] = []
        for segment in segments:
            holdout_size = min(max(2, len(segment) // 5), 6)
            if len(segment) <= holdout_size + 2:
                continue
            try:
                train_frame = segment.iloc[:-holdout_size].reset_index(drop=True)
                holdout_frame = segment.iloc[-holdout_size:].reset_index(drop=True)
                segment_series = dataclass_replace(series, frame=train_frame)

                adapter = self._registry.create(spec, params)
                adapter.initialize()
                adapter.train(segment_series)

                actual = holdout_frame[series.target_column].astype(float).to_numpy()
                vector = self._permute(adapter, holdout_frame, actual, feature_columns, seed=7)
                if vector:
                    vectors.append(vector)
            except Exception:  # noqa: BLE001 - one bad segment is simply dropped
                continue
        return vectors

    # Importance recomputed against a shorter near-term slice of the same
    # holdout — stability across forecasting horizons
    def _horizon_permutation_importance(
        self,
        adapter: BaseForecastingModel,
        holdout_frame: pd.DataFrame,
        actual: np.ndarray,
        feature_columns: tuple[str, ...],
        point_importance: ImportanceVector,
    ) -> list[ImportanceVector]:
        vectors: list[ImportanceVector] = []
        for index, fraction in enumerate(self._config.horizon_fractions):
            if fraction >= 0.999:
                vectors.append(point_importance)
                continue
            size = max(1, int(len(holdout_frame) * fraction))
            sub_holdout = holdout_frame.head(size)
            sub_actual = actual[:size]
            if len(sub_holdout) < 1:
                continue
            vector = self._permute(adapter, sub_holdout, sub_actual, feature_columns, seed=500 + index)
            if vector:
                vectors.append(vector)
        return vectors

    # Permute each feature column and score the resulting WMAPE degradation
    def _permute(
        self,
        adapter: BaseForecastingModel,
        holdout_frame: pd.DataFrame,
        actual: np.ndarray,
        feature_columns: tuple[str, ...],
        seed: int,
    ) -> ImportanceVector:
        rng = np.random.default_rng(seed)
        horizon = len(holdout_frame)

        baseline_output = adapter.predict(horizon, future_frame=holdout_frame)
        baseline_metrics = self._calculator.calculate(actual, baseline_output.values[:horizon])
        baseline_score = baseline_metrics.wmape if baseline_metrics.wmape is not None else 0.0

        importances: ImportanceVector = {}
        for column in feature_columns:
            shuffled = holdout_frame.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            output = adapter.predict(horizon, future_frame=shuffled)
            metrics = self._calculator.calculate(actual, output.values[:horizon])
            score = metrics.wmape if metrics.wmape is not None else baseline_score
            importances[column] = max(0.0, score - baseline_score)

        return _normalize(importances)

    # ------------------------------------------------------------------
    # Shared assembly
    # ------------------------------------------------------------------

    # Fraction of importance weight resting on features that correlate with the target
    def _plausibility(
        self, importance: ImportanceVector, feature_frame: pd.DataFrame, series: ForecastSeries
    ) -> float:
        target = series.frame[series.target_column].astype(float).to_numpy()

        plausible_weight = 0.0
        total_weight = 0.0
        for feature, weight in importance.items():
            total_weight += weight
            column = feature_frame[feature].to_numpy(dtype=float) if feature in feature_frame.columns else None
            if column is None or len(column) != len(target):
                continue
            if np.std(column) == 0 or np.std(target) == 0:
                continue
            correlation = float(np.corrcoef(column, target)[0, 1])
            if np.isfinite(correlation) and abs(correlation) >= self._config.min_correlation_for_plausibility:
                plausible_weight += weight

        if total_weight == 0:
            return 1.0
        return plausible_weight / total_weight

    # Combine point importance and all perturbation vectors into the final result
    def _assemble(
        self,
        base: ExplainabilityResult,
        method: str,
        point_importance: ImportanceVector,
        local_importance: list[dict[str, Any]],
        shap_values_available: bool,
        series: ForecastSeries,
        feature_frame: pd.DataFrame | None,
        bootstrap_vectors: list[ImportanceVector],
        window_vectors: list[ImportanceVector],
        horizon_vectors: list[ImportanceVector],
    ) -> ExplainabilityResult:
        plausibility_frame = feature_frame if feature_frame is not None else series.frame
        plausibility = self._plausibility(point_importance, plausibility_frame, series)

        window_stability = _vector_consistency(bootstrap_vectors + window_vectors)
        horizon_stability = _vector_consistency(horizon_vectors)

        all_vectors = bootstrap_vectors + window_vectors + horizon_vectors
        dominant_feature, dominant_consistency, significant_changes = _feature_change_analysis(
            point_importance, all_vectors, self._config
        )

        weights = self._config
        components = []
        weight_total = 0.0
        if window_stability is not None:
            components.append(window_stability * weights.stability_weight)
            weight_total += weights.stability_weight
        components.append(plausibility * weights.plausibility_weight)
        weight_total += weights.plausibility_weight
        if horizon_stability is not None:
            components.append(horizon_stability * weights.window_consistency_weight)
            weight_total += weights.window_consistency_weight

        composite = sum(components) / weight_total if weight_total else plausibility

        base.method = method
        base.global_importance = point_importance
        base.local_importance = local_importance
        base.shap_values_available = shap_values_available
        base.stability = StabilityMetrics(
            window_stability=window_stability,
            horizon_stability=horizon_stability,
            plausibility=plausibility,
            dominant_feature=dominant_feature,
            dominant_feature_consistency=dominant_consistency,
            significant_changes=significant_changes,
            composite_score=composite,
        )
        return base


# ----------------------------------------------------------------------
# Importance extractors
# ----------------------------------------------------------------------


# Native `.feature_importances_` — the fallback when `shap` is not installed
def _native_importance(estimator: Any, features: pd.DataFrame) -> ImportanceVector:
    raw = getattr(estimator, "feature_importances_", None)
    if raw is None:
        return {}
    return _normalize({name: float(value) for name, value in zip(features.columns, raw)})


# Mean absolute SHAP value per feature, via TreeExplainer
def _shap_importance(estimator: Any, features: pd.DataFrame) -> ImportanceVector:
    importance, _ = _shap_importance_and_values(estimator, features)
    return importance


# SHAP global importance and the raw per-row values behind it, computed once
# and shared so global and local importance never disagree
def _shap_importance_and_values(estimator: Any, features: pd.DataFrame) -> tuple[ImportanceVector, np.ndarray]:
    import shap  # noqa: PLC0415 - optional dependency, imported only when available

    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(features)
    if isinstance(values, list):  # some SHAP versions return a list per output
        values = values[0]
    values = np.asarray(values)
    mean_abs = np.mean(np.abs(values), axis=0)
    return _normalize({name: float(value) for name, value in zip(features.columns, mean_abs)}), values


# Per-observation SHAP attributions for the most recent `max_rows` rows
def _local_importance_records(
    raw_values: np.ndarray, features: pd.DataFrame, series: ForecastSeries, max_rows: int
) -> list[dict[str, Any]]:
    n = raw_values.shape[0]
    take = min(max_rows, n)
    start = n - take

    dates = None
    if series.date_column in series.frame.columns:
        dates = series.frame[series.date_column].reset_index(drop=True)

    records = []
    for row in range(start, n):
        record: dict[str, Any] = {
            features.columns[col]: round(float(raw_values[row, col]), 6) for col in range(features.shape[1])
        }
        if dates is not None and row < len(dates):
            record["date"] = str(dates.iloc[row])
        records.append(record)
    return records


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


# Numeric, declared regressor columns this model could plausibly use
def _usable_permutation_features(spec, series: ForecastSeries) -> tuple[str, ...]:
    if not spec.supports_features:
        return ()

    return tuple(
        column
        for column in series.feature_columns
        if column in series.frame.columns and pd.api.types.is_numeric_dtype(series.frame[column])
    )


# Split `frame` into `n_segments` contiguous, non-overlapping chunks
def _chronological_segments(frame: pd.DataFrame, n_segments: int, min_observations: int) -> list[pd.DataFrame]:
    n = len(frame)
    if n_segments < 2 or n < min_observations * 2:
        return []

    size = n // n_segments
    if size < min_observations:
        return []

    segments = []
    for i in range(n_segments):
        start = i * size
        end = n if i == n_segments - 1 else start + size
        segment = frame.iloc[start:end].reset_index(drop=True)
        if len(segment) >= min_observations:
            segments.append(segment)
    return segments


# Rescale a vector of non-negative weights to sum to 1.0
def _normalize(vector: ImportanceVector) -> ImportanceVector:
    total = sum(vector.values())
    if total <= 0:
        return dict(vector)
    return {name: value / total for name, value in vector.items()}


# Dominant feature, its cross-perturbation consistency, and features that moved significantly
def _feature_change_analysis(
    point: ImportanceVector, comparison_vectors: list[ImportanceVector], config: ExplainabilityConfig
) -> tuple[str | None, float | None, list[str]]:
    if not point:
        return None, None, []

    dominant = max(point, key=point.get)
    usable = [vector for vector in comparison_vectors if vector]

    matches = sum(1 for vector in usable if max(vector, key=vector.get) == dominant)
    consistency = matches / len(usable) if usable else None

    changed: set[str] = set()
    for vector in usable:
        for feature, point_share in point.items():
            if abs(vector.get(feature, 0.0) - point_share) > config.significant_change_threshold:
                changed.add(feature)

    return dominant, consistency, sorted(changed)


# Average pairwise rank agreement across a set of importance vectors
def _vector_consistency(vectors: list[ImportanceVector]) -> float | None:
    usable = [v for v in vectors if v]
    if len(usable) < 2:
        return None

    keys = sorted(set().union(*[set(v.keys()) for v in usable]))
    if len(keys) < 2:
        return 1.0

    arrays = [np.array([v.get(key, 0.0) for key in keys]) for v in usable]

    correlations = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            correlation = _spearman(arrays[i], arrays[j])
            if correlation is not None:
                correlations.append(correlation)

    if not correlations:
        return None

    # Correlation lives in [-1, 1]; rescale to [0, 1] so it combines cleanly
    # with the other [0, 1] sub-scores.
    mean_correlation = float(np.mean(correlations))
    return max(0.0, min(1.0, (mean_correlation + 1.0) / 2.0))


# Spearman rank correlation between two importance vectors
def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0 if np.allclose(a, b) else None
    rank_a = pd.Series(a).rank().to_numpy()
    rank_b = pd.Series(b).rank().to_numpy()
    correlation = float(np.corrcoef(rank_a, rank_b)[0, 1])
    return correlation if np.isfinite(correlation) else None
