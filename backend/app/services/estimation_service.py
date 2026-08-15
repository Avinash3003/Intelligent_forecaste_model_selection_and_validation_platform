"""Estimates how long a run will take and what it will cost, before it runs.

Every number comes from the real uploaded dataset (rows, per-key history,
missingness) and the real selected configuration (models, horizon). No
forecasting code runs here.

Timings are calibrated from this deployment's own completed runs when there
are enough of them, otherwise from labelled heuristics — calibration_basis
on the response always says which was used.

The answer is a range, never one number: per-key runtime depends on series
length and on how many keys clear each model's minimum-history bar.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import pandas as pd

from app.config.settings import Settings, get_settings
from app.orchestration.mlflow_history import MLflowHistoryStore
from app.schemas.estimation import (
    CostBreakdown,
    DatasetMetadataSummary,
    EstimateComponent,
    EstimationRequest,
    EstimationResponse,
    WorkloadEstimate,
)
from app.services.frequency_detector import FrequencyDetector

# ---------------------------------------------------------------------
# Heuristic fallback constants — used only when historical telemetry is
# insufficient to calibrate from. Every one of these mirrors a real
# forecast_engine default so the two never drift silently apart; each is
# named after the config value it mirrors.
# ---------------------------------------------------------------------

# Mirrors forecast_engine/config/evaluation_config.py's BacktestConfig
# defaults (rolling strategy).
_BACKTEST_MIN_TRAIN_SIZE = 12
_BACKTEST_HORIZON = 3
_BACKTEST_MAX_WINDOWS = 5

# Mirrors ModelConfig.registry's per-model `min_observations` (model_config.py).
_MODEL_MIN_OBSERVATIONS: dict[str, int] = {
    "seasonal_naive": 2,
    # arima=20 / prophet=24, matching model_config.py exactly. These two
    # were previously transposed here, which mis-counted how many keys clear
    # each model's history bar and so skewed the workload the estimate is
    # built from.
    "arima": 20,
    "prophet": 24,
    "lightgbm": 24,
    "xgboost": 24,
    "tft": 60,
}
_DEFAULT_MIN_OBSERVATIONS = 24

# Mirrors TuningConfig.min_observations_for_tuning.
_TUNING_MIN_OBSERVATIONS = 48

# Seconds per model fit (training fit and backtest fold fit are treated as
# the same unit of work — both are "fit this model family once"), by
# family. Relative weights measured from this platform's own local runs;
# Prophet and TFT genuinely dominate. Used only when no historical
# calibration is available.
_HEURISTIC_SECONDS_PER_FIT: dict[str, float] = {
    "seasonal_naive": 0.02,
    "arima": 0.9,
    "prophet": 1.6,
    "lightgbm": 0.3,
    "xgboost": 0.35,
    "tft": 6.0,
}
_DEFAULT_SECONDS_PER_FIT = 0.75

_HEURISTIC_SECONDS_PER_SHAP = 0.5

# Measured from a real run of this platform against Azure OpenAI
# (gpt-4o-mini-class deployment, the v2 structured-insight prompt): ~1,090
# prompt tokens and ~104 completion tokens per group, ~2.0s latency.
_HEURISTIC_PROMPT_TOKENS_PER_CALL = 1100.0
_HEURISTIC_COMPLETION_TOKENS_PER_CALL = 110.0
_HEURISTIC_SECONDS_PER_LLM_CALL = 2.5

# Fixed per-run overhead not proportional to keys or models: load, quality
# assessment, preprocessing, curated write-back, ranking, MLflow logging.
_FIXED_OVERHEAD_SECONDS = 20.0

# Cloud execution waits for compute before any Python of ours runs; local
# execution has no such wait. Serverless is the primary cloud path and
# acquires compute in seconds, not minutes — charging it the same 5-minute
# classic-cluster figure added ~5 min of pure fiction to every cloud
# estimate, which is most of why estimates read far longer than the run
# actually took. Each mode now carries its own figure, and both are
# superseded by `_calibrate_startup_seconds()` as soon as this deployment
# has real completed runs to measure instead.
_SERVERLESS_STARTUP_SECONDS = 45.0
_CLASSIC_CLUSTER_STARTUP_SECONDS = 300.0

# Startup is only calibrated from runs whose measured overhead is credible:
# a negative or absurd gap means the two clocks disagree, not that startup
# was instant, so such a sample is discarded rather than averaged in.
_MAX_CREDIBLE_STARTUP_SECONDS = 900.0

# The estimate's uncertainty band around the calibrated/heuristic center.
_LOW_FACTOR = 0.7
_HIGH_FACTOR = 1.6

# How many recent completed runs to sample for calibration, and the
# minimum usable sample before "historical" replaces "heuristic".
_HISTORY_SAMPLE_SIZE = 15
_MIN_RUNS_FOR_CALIBRATION = 3


@dataclass
class _Calibration:
    seconds_per_fit: float
    seconds_per_shap: float
    seconds_per_llm_call: float
    prompt_tokens_per_call: float
    completion_tokens_per_call: float
    basis: str  # "heuristic" or "historical (N runs)"


class EstimationService:
    """Estimates runtime and compute/LLM cost for a configuration, without running it."""

    def __init__(self, settings: Settings | None = None, history: MLflowHistoryStore | None = None) -> None:
        self._settings = settings or get_settings()
        self._history = history or MLflowHistoryStore(self._settings)
        self._frequency_detector = FrequencyDetector()

    def estimate(self, dataframe: pd.DataFrame, request: EstimationRequest) -> EstimationResponse:
        dataset = self._dataset_summary(dataframe, request)
        period_counts = self._period_counts(dataframe, request, dataset)
        workload = self._workload(period_counts, dataset)
        calibration = self._calibrate()

        runtime_seconds = self._estimate_runtime_seconds(period_counts, dataset, workload, calibration)
        backend = (self._settings.execution_mode or "local").strip().lower()
        startup_seconds = self._startup_seconds(backend)

        low_minutes = (startup_seconds + runtime_seconds * _LOW_FACTOR) / 60.0
        high_minutes = (startup_seconds + runtime_seconds * _HIGH_FACTOR) / 60.0

        cost = self._estimate_cost(workload, low_minutes, high_minutes, calibration, backend)

        breakdown = [
            EstimateComponent(label="Dataset", detail=f"{dataset.rows:,} rows, {dataset.columns} columns"),
            EstimateComponent(
                label="Forecast groups",
                detail=f"{workload.forecast_groups:,} series" if workload.forecast_groups > 1 else "1 series (single-series dataset)",
            ),
            EstimateComponent(
                label="Models per group",
                detail=f"{workload.models_per_group} ({', '.join(dataset.selected_models)})"
                if dataset.selected_models
                else "all registered models",
            ),
            EstimateComponent(label="Horizon", detail=f"{request.horizon} months"),
            EstimateComponent(
                label="Model evaluations",
                detail=f"{workload.model_evaluations:,} (keys × models)",
            ),
            EstimateComponent(
                label="Backtest windows",
                detail=f"{workload.backtest_windows:,} fold(s) — the largest share of Evaluation time",
            ),
            EstimateComponent(
                label="LLM explanation calls",
                detail=f"{workload.llm_calls:,} (one per forecast group)",
            ),
        ]
        if startup_seconds:
            # Rendered in whichever unit reads honestly: a 45-second
            # serverless start is "~45 sec", not the "~0 min" an
            # unconditional minutes conversion produced.
            if startup_seconds < 90:
                startup_detail = f"~{int(round(startup_seconds))} sec before the pipeline begins"
            else:
                startup_detail = f"~{startup_seconds / 60:.0f} min before the pipeline begins"
            breakdown.append(EstimateComponent(label="Compute startup", detail=startup_detail))

        return EstimationResponse(
            dataset=dataset,
            workload=workload,
            cost=cost,
            estimated_minutes_low=round(low_minutes, 1),
            estimated_minutes_high=round(high_minutes, 1),
            estimated_duration_label=_duration_label(low_minutes, high_minutes),
            execution_backend=backend,
            breakdown=breakdown,
            basis=(
                f"Estimated from this dataset's actual shape and per-key history, and this run's "
                f"selected models and horizon, calibrated using {calibration.basis}. Actual runtime "
                f"varies with how many series have enough history to train every model."
            ),
            calibration_basis=calibration.basis,
        )

    # ------------------------------------------------------------------
    # Dataset metadata (Objective 2: "must use the ACTUAL uploaded dataset")
    # ------------------------------------------------------------------

    def _dataset_summary(self, dataframe: pd.DataFrame, request: EstimationRequest) -> DatasetMetadataSummary:
        meta = request.metadata
        key_columns = [c for c in meta.key_columns if c in dataframe.columns]
        feature_columns = [c for c in meta.feature_columns if c in dataframe.columns]

        grain = "Unknown"
        if meta.date_column in dataframe.columns:
            grain = self._frequency_detector.detect(dataframe[meta.date_column])

        periods_by_group = self._periods_by_group(dataframe, meta.date_column, key_columns)
        history_length = int(max(periods_by_group.values(), default=0))

        missingness = None
        if meta.target_column in dataframe.columns:
            target = pd.to_numeric(dataframe[meta.target_column], errors="coerce")
            missingness = round(float(target.isna().mean() * 100.0), 2)

        return DatasetMetadataSummary(
            rows=int(dataframe.shape[0]),
            columns=int(dataframe.shape[1]),
            date_column=meta.date_column,
            target_column=meta.target_column,
            feature_columns=feature_columns,
            key_columns=key_columns,
            unique_keys=len(periods_by_group) or 1,
            date_grain=grain,
            history_length_periods=history_length,
            missingness_pct=missingness,
            forecast_horizon=request.horizon,
            selected_models=self._model_ids(request),
        )

    def _periods_by_group(
        self, dataframe: pd.DataFrame, date_column: str, key_columns: list[str]
    ) -> dict[tuple, int]:
        """Distinct months per group — what the curated dataset will hold after
        monthly aggregation, and a better proxy than raw row count."""
        if date_column not in dataframe.columns:
            return {}

        dates = pd.to_datetime(dataframe[date_column], errors="coerce")
        periods = dates.dt.to_period("M")

        if not key_columns:
            valid = periods.dropna()
            return {(): int(valid.nunique())} if len(valid) else {}

        try:
            grouped = periods.groupby([dataframe[c] for c in key_columns]).nunique()
        except (TypeError, ValueError):
            return {}
        return {tuple([key]) if not isinstance(key, tuple) else key: int(count) for key, count in grouped.items()}

    # ------------------------------------------------------------------
    # Workload (Objective 2: "derived from the actual workload", not keys x models alone)
    # ------------------------------------------------------------------

    def _period_counts(
        self, dataframe: pd.DataFrame, request: EstimationRequest, dataset: DatasetMetadataSummary
    ) -> list[int]:
        periods_by_group = self._periods_by_group(
            dataframe, request.metadata.date_column, [c for c in request.metadata.key_columns if c in dataframe.columns]
        )
        return list(periods_by_group.values()) or [dataset.history_length_periods]

    def _workload(self, period_counts: list[int], dataset: DatasetMetadataSummary) -> WorkloadEstimate:
        models = dataset.selected_models
        groups = dataset.unique_keys
        model_evaluations = groups * len(models)

        backtest_windows = sum(
            _backtest_window_count(periods) for periods in period_counts for _ in models
        )

        trained_pairs = sum(
            1
            for periods in period_counts
            for model in models
            if periods >= _MODEL_MIN_OBSERVATIONS.get(model, _DEFAULT_MIN_OBSERVATIONS)
        )

        tuning_eligible = sum(
            1
            for periods in period_counts
            for model in models
            if periods >= _TUNING_MIN_OBSERVATIONS and model != "seasonal_naive"
        )

        return WorkloadEstimate(
            forecast_groups=groups,
            models_per_group=len(models),
            model_evaluations=model_evaluations,
            backtest_windows=backtest_windows,
            # Every trained pair is forward-validated (Section 6.5 runs
            # regardless of the backtest's own outcome) and reuses
            # Training's own fit rather than a second one — see
            # forecast_engine's ForwardForecastGenerator.
            forward_validation_forecasts=trained_pairs,
            tuning_eligible_pairs=tuning_eligible,
            # Upper bound: SHAP runs only for models that survive forward
            # validation, and survival depends on the actual forecast, not
            # just the dataset's shape — not knowable before the run.
            shap_computations=trained_pairs,
            llm_calls=groups,
        )

    def _model_ids(self, request: EstimationRequest) -> list[str]:
        """Selected models, or the whole registry when none were chosen —
        an empty selection means "train everything" to the engine."""
        selected = [model.strip().lower() for model in request.selected_models if model.strip()]
        return selected or sorted(_MODEL_MIN_OBSERVATIONS)

    # ------------------------------------------------------------------
    # Historical calibration (Section 8.5.1)
    # ------------------------------------------------------------------

    def _startup_seconds(self, backend: str) -> float:
        """Seconds spent waiting for compute before stage one.

        Zero locally. In the cloud it is measured from real runs — wall-clock
        duration minus the engine's own stage time is exactly the time spent
        outside the engine — falling back to the per-mode constant only when
        no credible sample exists.
        """
        if backend not in ("databricks", "databricks_dcs"):
            return 0.0

        default = (
            _CLASSIC_CLUSTER_STARTUP_SECONDS
            if backend == "databricks_dcs"
            else _SERVERLESS_STARTUP_SECONDS
        )

        measured = self._measured_startup_seconds()
        return measured if measured is not None else default

    def _measured_startup_seconds(self) -> float | None:
        """Median non-engine overhead across recent completed runs, or None."""
        samples: list[float] = []
        try:
            listings = self._history.list_runs(limit=_HISTORY_SAMPLE_SIZE)
        except Exception:  # noqa: BLE001 - calibration is best-effort
            return None

        for listing in listings:
            if listing.job_status.value != "Completed" or not listing.duration_seconds:
                continue
            try:
                summary = self._history.get_summary(listing.run_id)
            except Exception:  # noqa: BLE001
                continue
            if not summary:
                continue

            stage_total = _total_stage_seconds(summary)
            if stage_total is None:
                continue

            overhead = float(listing.duration_seconds) - stage_total
            if 0.0 <= overhead <= _MAX_CREDIBLE_STARTUP_SECONDS:
                samples.append(overhead)

        if len(samples) < _MIN_RUNS_FOR_CALIBRATION:
            return None
        return statistics.median(samples)

    def _calibrate(self) -> _Calibration:
        """Per-fit and per-call timings measured from recent completed runs.

        Runs whose summary predates the timing breakdown are skipped — there
        is nothing to calibrate from data that was never recorded.
        """
        fit_ratios: list[float] = []
        shap_ratios: list[float] = []
        llm_latencies: list[float] = []
        prompt_tokens: list[float] = []
        completion_tokens: list[float] = []
        # Counted separately from `fit_ratios`: a single run can contribute
        # up to two fit-ratio samples (backtest and training), so using
        # len(fit_ratios) as "how many runs was this calibrated from" would
        # both overstate the run count in the label and let two ratios
        # from one single run satisfy the minimum-run gate below.
        runs_used = 0

        try:
            listings = self._history.list_runs(limit=_HISTORY_SAMPLE_SIZE)
        except Exception:  # noqa: BLE001 - calibration is best-effort
            listings = []

        for listing in listings:
            if listing.job_status.value != "Completed":
                continue
            try:
                summary = self._history.get_summary(listing.run_id)
            except Exception:  # noqa: BLE001
                summary = None
            if not summary:
                continue

            evaluation = summary.get("evaluation_report") or {}
            timing = evaluation.get("timing_breakdown") or {}
            backtest_seconds = timing.get("backtest_seconds")
            backtest_windows = timing.get("backtest_windows_evaluated")
            used_this_run = False
            if backtest_seconds and backtest_windows:
                fit_ratios.append(backtest_seconds / backtest_windows)
                used_this_run = True

            training = summary.get("training_report") or {}
            train_seconds = training.get("duration_seconds")
            trained_count = training.get("trained")
            if train_seconds and trained_count:
                fit_ratios.append(train_seconds / trained_count)
                used_this_run = True

            if used_this_run:
                runs_used += 1

            explainability = _stage_duration(summary, "Generate Explainability (SHAP)")
            explainability_report = summary.get("explainability_report") or {}
            surviving = len(explainability_report.get("results") or [])
            if explainability and surviving:
                shap_ratios.append(explainability / surviving)

            insight = summary.get("insight_report") or {}
            trace = insight.get("trace_summary") or {}
            if trace.get("call_count") and trace.get("average_latency_ms"):
                llm_latencies.append(trace["average_latency_ms"] / 1000.0)
                calls = trace["call_count"]
                if trace.get("prompt_tokens"):
                    prompt_tokens.append(trace["prompt_tokens"] / calls)
                if trace.get("completion_tokens"):
                    completion_tokens.append(trace["completion_tokens"] / calls)

        if fit_ratios and runs_used >= _MIN_RUNS_FOR_CALIBRATION:
            return _Calibration(
                seconds_per_fit=statistics.median(fit_ratios),
                seconds_per_shap=statistics.median(shap_ratios) if shap_ratios else _HEURISTIC_SECONDS_PER_SHAP,
                seconds_per_llm_call=statistics.median(llm_latencies) if llm_latencies else _HEURISTIC_SECONDS_PER_LLM_CALL,
                prompt_tokens_per_call=statistics.median(prompt_tokens) if prompt_tokens else _HEURISTIC_PROMPT_TOKENS_PER_CALL,
                completion_tokens_per_call=statistics.median(completion_tokens) if completion_tokens else _HEURISTIC_COMPLETION_TOKENS_PER_CALL,
                basis=f"historical telemetry ({runs_used} recent completed run(s))",
            )

        return _Calibration(
            seconds_per_fit=_DEFAULT_SECONDS_PER_FIT,
            seconds_per_shap=_HEURISTIC_SECONDS_PER_SHAP,
            seconds_per_llm_call=_HEURISTIC_SECONDS_PER_LLM_CALL,
            prompt_tokens_per_call=_HEURISTIC_PROMPT_TOKENS_PER_CALL,
            completion_tokens_per_call=_HEURISTIC_COMPLETION_TOKENS_PER_CALL,
            basis="a measured-timing heuristic (not enough completed runs yet for historical calibration)",
        )

    # ------------------------------------------------------------------
    # Runtime + cost
    # ------------------------------------------------------------------

    def _estimate_runtime_seconds(
        self,
        period_counts: list[int],
        dataset: DatasetMetadataSummary,
        workload: WorkloadEstimate,
        calibration: _Calibration,
    ) -> float:
        """Fit-seconds, weighted by model family.

        seconds_per_fit is one platform-wide average, so a TFT-only run must
        not be estimated as cheaply as a seasonal_naive-only one. Each model's
        heuristic weight rescales that average toward its real relative cost,
        which needs far fewer sampled runs than per-model calibration would.
        """
        fit_seconds = 0.0
        for model in dataset.selected_models:
            weight = _HEURISTIC_SECONDS_PER_FIT.get(model, _DEFAULT_SECONDS_PER_FIT) / _DEFAULT_SECONDS_PER_FIT
            # One training fit per group, plus that model's own backtest
            # folds across every group (forward-forecast generation reuses
            # the training fit — Objective 3 — so it adds no fit here).
            model_fits = len(period_counts) + sum(_backtest_window_count(periods) for periods in period_counts)
            fit_seconds += model_fits * calibration.seconds_per_fit * weight

        shap_seconds = workload.shap_computations * calibration.seconds_per_shap
        llm_seconds = workload.llm_calls * calibration.seconds_per_llm_call
        return _FIXED_OVERHEAD_SECONDS + fit_seconds + shap_seconds + llm_seconds

    def _estimate_cost(
        self,
        workload: WorkloadEstimate,
        low_minutes: float,
        high_minutes: float,
        calibration: _Calibration,
        backend: str,
    ) -> CostBreakdown:
        currency = self._settings.compute_cost_currency
        databricks_low = databricks_high = None
        databricks_available = False

        # Only priced when running on Databricks (either cloud mode) — local
        # execution has no cluster cost.
        rate = self._settings.compute_cost_per_hour
        if backend in ("databricks", "databricks_dcs") and rate is not None and rate > 0:
            databricks_low = round(low_minutes / 60.0 * rate, 2)
            databricks_high = round(high_minutes / 60.0 * rate, 2)
            databricks_available = True

        llm_low = llm_high = None
        llm_available = False
        input_rate = self._settings.azure_openai_price_input_per_1k
        output_rate = self._settings.azure_openai_price_output_per_1k
        if input_rate is not None and output_rate is not None and workload.llm_calls:
            per_call = (calibration.prompt_tokens_per_call / 1000.0) * input_rate + (
                calibration.completion_tokens_per_call / 1000.0
            ) * output_rate
            base = workload.llm_calls * per_call
            llm_low = round(base * _LOW_FACTOR, 4)
            llm_high = round(base * _HIGH_FACTOR, 4)
            llm_available = True

        total_low = total_high = None
        total_available = False
        if databricks_available or llm_available:
            total_low = round((databricks_low or 0.0) + (llm_low or 0.0), 2)
            total_high = round((databricks_high or 0.0) + (llm_high or 0.0), 2)
            total_available = True

        return CostBreakdown(
            databricks_cost_low=databricks_low,
            databricks_cost_high=databricks_high,
            databricks_cost_available=databricks_available,
            llm_cost_low=llm_low,
            llm_cost_high=llm_high,
            llm_cost_available=llm_available,
            total_cost_low=total_low,
            total_cost_high=total_high,
            total_cost_available=total_available,
            currency=currency,
        )


def _backtest_window_count(
    history_periods: int,
    min_train: int = _BACKTEST_MIN_TRAIN_SIZE,
    horizon: int = _BACKTEST_HORIZON,
    max_windows: int = _BACKTEST_MAX_WINDOWS,
) -> int:
    """The engine's rolling-backtest fold count, duplicated rather than
    imported — the two run in separate virtual environments."""
    if history_periods < min_train + horizon:
        return 0
    windows = 0
    train_end = min_train
    step = horizon
    while train_end + horizon <= history_periods and windows < max_windows:
        windows += 1
        train_end += step
    return windows


def _stage_duration(summary: dict, stage_name: str) -> float | None:
    for stage in summary.get("stages") or []:
        if stage.get("name") == stage_name:
            return stage.get("duration_seconds")
    return None


def _total_stage_seconds(summary: dict) -> float | None:
    """Seconds the engine accounted for across its stages, or None if nothing
    was measured — treating that as zero would read the whole run as startup."""
    durations = [
        stage.get("duration_seconds")
        for stage in summary.get("stages") or []
        if isinstance(stage.get("duration_seconds"), (int, float))
    ]
    return float(sum(durations)) if durations else None


def _duration_label(low_minutes: float, high_minutes: float) -> str:
    if high_minutes < 1:
        return "under a minute"
    if high_minutes < 60:
        return f"{max(int(math.floor(low_minutes)), 1)}–{int(math.ceil(high_minutes))} min"
    return f"{low_minutes / 60:.1f}–{high_minutes / 60:.1f} hours"


_service: EstimationService | None = None


def get_estimation_service() -> EstimationService:
    global _service
    if _service is None:
        _service = EstimationService()
    return _service
