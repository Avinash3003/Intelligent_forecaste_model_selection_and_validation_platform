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

import logging
import math
import statistics
import threading
import time
from dataclasses import dataclass

import pandas as pd

from app.config.model_availability import (
    filter_available,
    unavailable_models,
)
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
from app.services.dataset_analysis import DatasetAnalysis, DatasetAnalyzer
from app.services.pipeline_stages import canonical_stage_name

logger = logging.getLogger(__name__)

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
# execution has no such wait.
#
# A run's seven tasks all attach to ONE cluster (the shared job cluster —
# see backend/app/orchestration/databricks_runner.py), either an existing
# cluster (starting on demand if it was TERMINATED) or a freshly provisioned
# job cluster. Either way the run pays one cold start, not seven — this
# fallback is sized from two real cold starts observed on this project's own
# workspace: 351s (a TERMINATED all-purpose cluster starting on demand) and
# 441s (a job cluster provisioning from nothing).
#
# This figure is a fallback only. `_calibrate_startup_seconds()` measures
# wall-clock minus engine time from real history and takes over as soon as
# this deployment has completed runs to learn from.
_CLOUD_STARTUP_SECONDS = 396.0  # midpoint of the two observed cold starts

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

# How many run summaries the sweep will actually download. Each one is a
# remote artifact fetch on Databricks, and calibration only ever takes a
# *median* — past a handful of samples another download buys precision
# nobody can perceive while costing a full round trip. Scanning still walks
# up to _HISTORY_SAMPLE_SIZE listings, so runs that turn out to have no
# summary don't consume the budget; only successful fetches do.
_CALIBRATION_SAMPLE_TARGET = 6

# How long a computed calibration is reused before the sweep runs again.
# Calibration only moves when new runs complete, and a forecasting run
# takes minutes to hours — so re-deriving it per /estimate call re-paid a
# large network cost for an answer that had not changed. Five minutes is
# far shorter than the interval between completed runs, so a genuinely new
# run is picked up long before it could matter to an estimate.
_CALIBRATION_TTL_SECONDS = 300.0

# Hard wall-clock ceiling on one history sweep.
#
# Calibration is an ACCURACY improvement over the heuristic, never a
# correctness requirement — `_sweep_run_history` already degrades to the
# heuristic on every failure mode (MLflow unreachable, no runs, no summary
# artifact). A slow sweep is just one more of those, and must be treated the
# same way instead of holding the request open.
#
# Measured against this deployment's real Databricks-backed MLflow store:
# `list_runs(15)` alone took 24.7s, and a single `get_summary()` then ran
# past 95s — with the sweep allowed up to _CALIBRATION_SAMPLE_TARGET (6) of
# them. That blew through the frontend's 30s request timeout and surfaced in
# the wizard as "The request took too long to respond" on the Estimate step,
# while the backend was still working. Bounding the sweep fixes that at the
# source rather than by widening the client timeout, which would only move
# a multi-minute wait into the UI.
_CALIBRATION_DEADLINE_SECONDS = 8.0


def _heuristic_history_calibration() -> "_RunHistoryCalibration":
    """The no-history answer: labelled heuristics, no measured startup.

    Identical to what `_sweep_run_history` produces when it finds nothing
    usable — defined separately so the request path can return it without
    performing the sweep's network I/O at all.
    """
    return _RunHistoryCalibration(
        calibration=_Calibration(
            seconds_per_fit=_DEFAULT_SECONDS_PER_FIT,
            seconds_per_shap=_HEURISTIC_SECONDS_PER_SHAP,
            seconds_per_llm_call=_HEURISTIC_SECONDS_PER_LLM_CALL,
            prompt_tokens_per_call=_HEURISTIC_PROMPT_TOKENS_PER_CALL,
            completion_tokens_per_call=_HEURISTIC_COMPLETION_TOKENS_PER_CALL,
            basis="a measured-timing heuristic (not enough completed runs yet for historical calibration)",
        ),
        measured_startup_seconds=None,
    )


@dataclass
class _Calibration:
    seconds_per_fit: float
    seconds_per_shap: float
    seconds_per_llm_call: float
    prompt_tokens_per_call: float
    completion_tokens_per_call: float
    basis: str  # "heuristic" or "historical (N runs)"


@dataclass(frozen=True)
class _RunHistoryCalibration:
    """Everything one sweep of MLflow run history yields.

    Both consumers — per-fit timings and measured compute startup — read the
    *same* recent runs and the *same* summary artifacts. Producing them
    together is what removes the second full history sweep (and its second
    round of artifact downloads) from every estimate.
    """

    calibration: _Calibration
    # Median non-engine overhead across the sampled runs; None when no
    # credible sample existed, which is what makes the caller fall back to
    # the per-mode constant.
    measured_startup_seconds: float | None


class EstimationService:
    """Estimates runtime and compute/LLM cost for a configuration, without running it."""

    def __init__(self, settings: Settings | None = None, history: MLflowHistoryStore | None = None) -> None:
        self._settings = settings or get_settings()
        self._history = history or MLflowHistoryStore(self._settings)
        self._analyzer = DatasetAnalyzer()
        # Guards the TTL cache below. This service is a process-wide
        # singleton (see get_estimation_service), so the cache is shared by
        # every concurrent request rather than per-request.
        self._calibration_lock = threading.Lock()
        self._cached_history: _RunHistoryCalibration | None = None
        # Guards against every concurrent estimate starting its own sweep
        # while the first one is still running.
        self._calibration_refreshing = False
        self._cached_history_at: float = 0.0

    def estimate(self, dataframe: pd.DataFrame, request: EstimationRequest) -> EstimationResponse:
        backend = (self._settings.execution_mode or "local").strip().lower()

        # One pass over the dataset. Everything below reads from `analysis`
        # rather than touching the DataFrame again.
        analysis = self._analyzer.analyze(dataframe, request.metadata)
        dataset = self._dataset_summary(analysis, request, backend)
        period_counts = analysis.period_counts

        workload = self._workload(period_counts, dataset)

        # One sweep of run history, shared by calibration and startup.
        history = self._run_history_calibration()
        calibration = history.calibration

        runtime_seconds = self._estimate_runtime_seconds(period_counts, dataset, workload, calibration)
        startup_seconds = self._startup_seconds(backend, history.measured_startup_seconds)

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
            # startup is "~45 sec", not the "~0 min" an unconditional
            # minutes conversion produced.
            if startup_seconds < 90:
                startup_detail = f"~{int(round(startup_seconds))} sec before the pipeline begins"
            else:
                startup_detail = f"~{startup_seconds / 60:.0f} min before the pipeline begins"
            breakdown.append(EstimateComponent(label="Compute startup", detail=startup_detail))

        # Named explicitly rather than silently dropped: a user who ticked
        # TFT and sees no TFT cost is owed the reason.
        excluded = self._excluded_models(request, backend)
        if excluded:
            breakdown.append(
                EstimateComponent(
                    label="Excluded models",
                    detail=f"{', '.join(sorted(excluded))} — not runnable on this execution mode, so not estimated",
                )
            )

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

    def _dataset_summary(
        self, analysis: DatasetAnalysis, request: EstimationRequest, backend: str
    ) -> DatasetMetadataSummary:
        """The response's dataset block, read entirely from the single
        analysis pass — this method touches no DataFrame."""
        return DatasetMetadataSummary(
            rows=analysis.rows,
            columns=analysis.columns,
            date_column=request.metadata.date_column,
            target_column=request.metadata.target_column,
            feature_columns=analysis.feature_columns,
            key_columns=analysis.key_columns,
            unique_keys=analysis.unique_keys,
            date_grain=analysis.date_grain,
            history_length_periods=analysis.history_length_periods,
            missingness_pct=analysis.missingness_pct,
            forecast_horizon=request.horizon,
            selected_models=self._model_ids(request, backend),
        )

    # ------------------------------------------------------------------
    # Workload (Objective 2: "derived from the actual workload", not keys x models alone)
    # ------------------------------------------------------------------

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

    def _model_ids(self, request: EstimationRequest, backend: str) -> list[str]:
        """Models this run will actually train.

        Selected models, or the whole registry when none were chosen (an
        empty selection means "train everything" to the engine), minus any
        the current execution mode cannot run. The availability filter is
        applied *after* the registry fallback so both paths are filtered,
        and charging runtime for a model the environment will report
        Unavailable is exactly the fiction it removes.
        """
        selected = [model.strip().lower() for model in request.selected_models if model.strip()]
        available = filter_available(selected or sorted(_MODEL_MIN_OBSERVATIONS), backend)
        # Also drop models that are offered but never executed (TFT — see
        # model_availability.SILENTLY_SKIPPED_MODELS). Charging runtime for
        # work that will not happen is the same fiction the availability
        # filter above exists to remove.
        return list(available)

    def _excluded_models(self, request: EstimationRequest, backend: str) -> list[str]:
        """Models the user asked for that this execution mode cannot run."""
        blocked = unavailable_models(backend)
        return [
            model.strip().lower()
            for model in request.selected_models
            if model.strip().lower() in blocked
        ]

    # ------------------------------------------------------------------
    # Historical calibration (Section 8.5.1)
    # ------------------------------------------------------------------

    def _startup_seconds(self, backend: str, measured: float | None) -> float:
        """Seconds spent waiting for compute before stage one.

        Zero locally. In the cloud `measured` is the real figure taken from
        run history — wall-clock duration minus the engine's own stage time
        is exactly the time spent outside the engine — and the per-mode
        constant is used only when no credible sample existed.

        The measurement is passed in rather than fetched here so it comes
        from the same history sweep calibration used.
        """
        if backend != "databricks":
            return 0.0

        return measured if measured is not None else _CLOUD_STARTUP_SECONDS

    def _run_history_calibration(self) -> _RunHistoryCalibration:
        """The cached history sweep, recomputed only once per TTL.

        Reused across requests because calibration is a property of this
        deployment's past runs, not of the estimate being asked for. Two
        estimates a second apart cannot have different history, so making
        the second one re-download the first one's artifacts was pure cost.
        """
        now = time.monotonic()
        with self._calibration_lock:
            cached = self._cached_history
            fresh = cached is not None and now - self._cached_history_at < _CALIBRATION_TTL_SECONDS
            if fresh:
                return cached
            already_refreshing = self._calibration_refreshing
            if not already_refreshing:
                self._calibration_refreshing = True

        # The sweep is never awaited on the request path. Against this
        # deployment's Databricks-backed store one sweep measured >120s
        # unbounded, and ~30s even with an internal deadline — because
        # `list_runs` alone took ~25s inside MLflow's own client, where no
        # caller-side deadline can interrupt it. Blocking /estimate on that
        # is what surfaced as "The request took too long to respond" on the
        # wizard's Estimate step.
        #
        # So: serve what we have (a stale sweep, or the heuristic on the
        # very first call) and refresh behind the request. Calibration only
        # sharpens an estimate that is already correct without it, so
        # answering now from the heuristic and sharpening the next call is
        # strictly better than making the user wait for precision.
        # The sweep runs on its own thread, and this request waits only up
        # to the deadline for it. A fast store (a local sqlite tracking
        # store, or an in-memory one) finishes well inside that and the
        # caller gets a fully calibrated answer on the very first estimate,
        # exactly as before. A slow one (Databricks, measured ~30s+) blows
        # the deadline, the caller gets the heuristic now, and the sweep
        # keeps going so the *next* estimate is calibrated.
        if not already_refreshing:
            worker = threading.Thread(
                target=self._refresh_calibration_in_background,
                name="estimation-calibration-refresh",
                daemon=True,
            )
            worker.start()
            worker.join(timeout=_CALIBRATION_DEADLINE_SECONDS)

        with self._calibration_lock:
            if self._cached_history is not None:
                return self._cached_history

        logger.warning(
            "Run-history calibration did not finish within %.0fs; estimating from the "
            "heuristic and refreshing in the background",
            _CALIBRATION_DEADLINE_SECONDS,
        )
        return _heuristic_history_calibration()

    def _refresh_calibration_in_background(self) -> None:
        """Recompute the sweep off the request path, then publish it.

        Failure is not propagated anywhere: the cache simply keeps its
        previous value (or stays empty, leaving the heuristic in place),
        which is the same degradation every other calibration failure mode
        already takes.
        """
        try:
            computed = self._sweep_run_history()
        except Exception:  # noqa: BLE001 - calibration is best-effort
            logger.warning("Background run-history calibration failed", exc_info=True)
            return
        finally:
            with self._calibration_lock:
                self._calibration_refreshing = False

        with self._calibration_lock:
            self._cached_history = computed
            self._cached_history_at = time.monotonic()

    def _sweep_run_history(self) -> _RunHistoryCalibration:
        """One pass over recent completed runs, yielding both calibrations.

        Runs whose summary predates the timing breakdown are skipped — there
        is nothing to calibrate from data that was never recorded. Every
        failure mode (MLflow unreachable, no runs, no summary artifact)
        degrades to the heuristic rather than raising.
        """
        fit_ratios: list[float] = []
        shap_ratios: list[float] = []
        llm_latencies: list[float] = []
        prompt_tokens: list[float] = []
        completion_tokens: list[float] = []
        startup_samples: list[float] = []
        # Counted separately from `fit_ratios`: a single run can contribute
        # up to two fit-ratio samples (backtest and training), so using
        # len(fit_ratios) as "how many runs was this calibrated from" would
        # both overstate the run count in the label and let two ratios
        # from one single run satisfy the minimum-run gate below.
        runs_used = 0
        summaries_read = 0

        # Every remote call below is bounded by one shared deadline, so a
        # slow store costs the estimate a known few seconds rather than an
        # unbounded wait. Whatever was gathered before the deadline is still
        # used; falling short only means falling back to the heuristic.
        deadline = time.monotonic() + _CALIBRATION_DEADLINE_SECONDS

        try:
            listings = self._history.list_runs(limit=_HISTORY_SAMPLE_SIZE)
        except Exception:  # noqa: BLE001 - calibration is best-effort
            listings = []

        # `list_runs` is itself a remote call and was measured at ~25s
        # against this deployment's store — on its own already past the
        # point where reading summaries could finish in time. Re-checking
        # here means a slow listing costs the estimate the deadline and
        # nothing more.
        if time.monotonic() >= deadline:
            logger.warning(
                "Run-history listing alone exceeded the %.0fs calibration deadline; "
                "estimating from the heuristic instead",
                _CALIBRATION_DEADLINE_SECONDS,
            )
            listings = []

        for listing in listings:
            if summaries_read >= _CALIBRATION_SAMPLE_TARGET:
                break
            if time.monotonic() >= deadline:
                logger.warning(
                    "Run-history calibration hit its %.0fs deadline after %d summary read(s); "
                    "estimating from the heuristic instead",
                    _CALIBRATION_DEADLINE_SECONDS,
                    summaries_read,
                )
                break
            if listing.job_status.value != "Completed":
                continue
            try:
                summary = self._history.get_summary(listing.run_id)
            except Exception:  # noqa: BLE001
                summary = None
            if not summary:
                continue
            summaries_read += 1

            # Compute startup, from the same summary rather than a second
            # fetch. getattr keeps a listing without a recorded duration
            # (or a stub in a test) from breaking the whole sweep.
            duration = getattr(listing, "duration_seconds", None)
            stage_total = _total_stage_seconds(summary)
            if duration and stage_total is not None:
                overhead = float(duration) - stage_total
                if 0.0 <= overhead <= _MAX_CREDIBLE_STARTUP_SECONDS:
                    startup_samples.append(overhead)

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

            explainability = _stage_duration(summary, "Explain Models")
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

        # Unchanged gate: startup is only reported as measured when enough
        # credible samples backed it, exactly as before.
        measured_startup = (
            statistics.median(startup_samples)
            if len(startup_samples) >= _MIN_RUNS_FOR_CALIBRATION
            else None
        )

        if fit_ratios and runs_used >= _MIN_RUNS_FOR_CALIBRATION:
            calibration = _Calibration(
                seconds_per_fit=statistics.median(fit_ratios),
                seconds_per_shap=statistics.median(shap_ratios) if shap_ratios else _HEURISTIC_SECONDS_PER_SHAP,
                seconds_per_llm_call=statistics.median(llm_latencies) if llm_latencies else _HEURISTIC_SECONDS_PER_LLM_CALL,
                prompt_tokens_per_call=statistics.median(prompt_tokens) if prompt_tokens else _HEURISTIC_PROMPT_TOKENS_PER_CALL,
                completion_tokens_per_call=statistics.median(completion_tokens) if completion_tokens else _HEURISTIC_COMPLETION_TOKENS_PER_CALL,
                basis=f"historical telemetry ({runs_used} recent completed run(s))",
            )
        else:
            calibration = _Calibration(
                seconds_per_fit=_DEFAULT_SECONDS_PER_FIT,
                seconds_per_shap=_HEURISTIC_SECONDS_PER_SHAP,
                seconds_per_llm_call=_HEURISTIC_SECONDS_PER_LLM_CALL,
                prompt_tokens_per_call=_HEURISTIC_PROMPT_TOKENS_PER_CALL,
                completion_tokens_per_call=_HEURISTIC_COMPLETION_TOKENS_PER_CALL,
                basis="a measured-timing heuristic (not enough completed runs yet for historical calibration)",
            )

        return _RunHistoryCalibration(calibration=calibration, measured_startup_seconds=measured_startup)

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

        # Only priced when running on Databricks — local execution has no
        # cluster cost.
        rate = self._settings.compute_cost_per_hour
        if backend == "databricks" and rate is not None and rate > 0:
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
    # Calibration reads historical summaries, which may predate the unified
    # stage vocabulary — match on the canonical name so a run recorded as
    # "Generate Explainability (SHAP)" still calibrates "Explain Models".
    for stage in summary.get("stages") or []:
        name = stage.get("name")
        if name and canonical_stage_name(name) == stage_name:
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
