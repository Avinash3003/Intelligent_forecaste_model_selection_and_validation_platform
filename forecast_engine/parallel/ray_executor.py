"""Runs the key-parallel workflow as four genuine, sequential stage
boundaries — train every key, then evaluate every key, then explain, then
rank and select — each one its own Ray fan-out.

    Stage: Train        -> N keys run train_key() concurrently -> aggregate
    Stage: Evaluate      -> N keys run evaluate_key() concurrently -> aggregate
    Stage: Explain       -> N keys run explain_key() concurrently -> aggregate
    Stage: Rank & Select -> N keys run rank_select_key() concurrently -> aggregate

Each stage waits for every key from the stage before it and starts nothing
until that dependency is satisfied — a key that failed an earlier stage is
dropped and takes no further part. Nothing inside one key ever crosses a
process boundary except at these four boundaries.

Falls back to running the same key_workflow calls in-process when Ray is
not installed, which keeps this importable everywhere and makes the
sequential and parallel paths provably the same code.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from forecast_engine.parallel.key_workflow import (
    KeyWorkflowConfig,
    evaluate_key,
    explain_key,
    rank_select_key,
    train_key,
)
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries
from forecast_engine.s04_training.model_trainer import TrainingReport
from forecast_engine.s05_models.base_model import TrainedModel
from forecast_engine.s06_evaluation.evaluation_report import EvaluationReport
from forecast_engine.s07_explainability.explainability_report import ExplainabilityReport
from forecast_engine.s08_ranking.ranking_report import RankingReport
from forecast_engine.s10_selection.selection_report import ProductionSelectionReport

logger = logging.getLogger(__name__)

# Fired after each task in the current stage completes: (stage_name, telemetry-so-far)
ProgressCallback = Callable[[str, dict[str, Any]], None]


# True when Ray can be imported in this environment
def ray_available() -> bool:
    try:
        import ray  # noqa: F401
    except ImportError:
        return False
    return True


# Copy the frame before a task touches it — never share buffers across
# tasks that might run on the same worker.
def _copy_series(series: ForecastSeries) -> ForecastSeries:
    return replace(series, frame=series.frame.copy())


# Spark worker nodes attached to this cluster, 0 on single-node compute
def _spark_worker_nodes() -> int:
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return 0
    spark = SparkSession.getActiveSession()
    if spark is None:
        return 0
    try:
        # One entry per executor plus the driver.
        executors = spark.sparkContext._jsc.sc().statusTracker().getExecutorInfos()
        return max(0, len(executors) - 1)
    except Exception:  # noqa: BLE001 - absence of executors is not an error
        return 0


def _start_ray() -> dict[str, Any]:
    """Start Ray across whatever compute this run actually has.

    Single-node compute keeps the local head; multi-node compute puts Ray
    on the Spark workers too, so the nodes the user paid for take keys
    instead of idling. Neither path fixes a CPU count — Ray reads the real
    resources either way, and each task still asks for exactly one CPU.

    Returns how Ray was actually started, so the run can REPORT it rather
    than leave it to be inferred. Falling back to the driver is a legitimate
    outcome on single-node compute and a silent, expensive defect on
    multi-node compute — a cluster whose workers were paid for and never
    used. The two are indistinguishable from the outside unless the
    decision is recorded, so this returns it and the stage executor puts it
    in the telemetry.
    """
    import ray

    runtime_env = _worker_runtime_env()
    workers = _spark_worker_nodes()
    topology: dict[str, Any] = {
        "spark_worker_nodes_detected": workers,
        "ray_on_spark_attempted": workers > 0,
        "ray_on_spark_error": None,
    }

    if workers > 0:
        try:
            from ray.util.spark import setup_ray_cluster

            setup_ray_cluster(max_worker_nodes=workers, collect_log_to_path=None)
            ray.init(ignore_reinit_error=True, address="auto", runtime_env=runtime_env)
            logger.info("Ray started across %d Spark worker node(s)", workers)
            topology["ray_mode"] = "ray_on_spark"
            return topology
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail the run
            # Loud, not silent: multi-node compute was requested and Ray
            # could not use it, so every key will run on the driver while
            # the worker nodes idle. That is a real, costly degradation and
            # it is recorded in the telemetry the run reports.
            logger.error(
                "Ray on Spark FAILED with %d Spark worker node(s) available (%s); "
                "falling back to driver-only Ray — worker nodes will be UNUSED",
                workers,
                exc,
            )
            topology["ray_on_spark_error"] = f"{type(exc).__name__}: {exc}"

    ray.init(ignore_reinit_error=True, include_dashboard=False, runtime_env=runtime_env)
    topology["ray_mode"] = "driver_only"
    return topology


_MATH_THREAD_VARS = (
    "OMP_NUM_THREADS",  # OpenMP — LightGBM, XGBoost, and BLAS backends
    "MKL_NUM_THREADS",  # Intel MKL
    "OPENBLAS_NUM_THREADS",  # OpenBLAS
    "NUMEXPR_NUM_THREADS",  # numexpr, which pandas uses for expressions
)


def _worker_runtime_env() -> dict[str, Any]:
    """The environment Ray's worker processes start with.

    Two things, both of which only matter in a worker:

    `PYTHONPATH` — a worker starts as a fresh interpreter and does not
    inherit the driver's `sys.path`, so a source checkout that the driver
    can import is invisible to it and every task fails to unpickle its own
    arguments. Putting the package's parent directory on the workers'
    PYTHONPATH covers that, and is a no-op where the wheel is properly
    installed (Databricks), since the directory is then already on the path.

    Thread caps — see `_MATH_THREAD_VARS`. These belong here rather than in
    the process environment because the driver is not a one-CPU process: it
    holds the whole node, and a sequential run there should still be free to
    use every core. Ray applies `runtime_env` variables when it starts the
    worker, which is before that worker imports numpy or statsmodels, and
    import time is when these libraries read them — setting them any later
    would have no effect.
    """
    engine_parent = str(Path(__file__).resolve().parents[2])
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if engine_parent not in parts:
        parts.insert(0, engine_parent)

    env_vars = {"PYTHONPATH": os.pathsep.join(parts)}

    env_vars.update({name: "1" for name in _MATH_THREAD_VARS})
    return {"env_vars": env_vars}


# A trained model's fitted estimator can hold state plain pickle refuses
# (TFT's cached network-output class — see
# TemporalFusionTransformerModel.prepare_for_pickling). Every trained
# model crosses a Ray boundary at least twice now — once returning from
# Train, again as Evaluate's input — so this runs right after training,
# before either crossing happens.
def _prepare_trained_models_for_pickling(report: TrainingReport) -> None:
    for result in report.trained_models():
        if result.fitted_model is not None:
            result.fitted_model.prepare_for_pickling()


# ---------------------------------------------------------------------
# Remote stage tasks — one per genuine stage, each asking for one CPU so
# Ray's own scheduler decides how many keys run at once.
# ---------------------------------------------------------------------


# Every payload that carries a fitted estimator crosses Ray as pre-pickled
# bytes, not a plain Python object -- Ray's own automatic serialization
# shares object-store memory as a READ-ONLY buffer, which a Cython/C
# state-space object (ARIMA's, notably) refuses to accept as its own
# writable state. Manual pickle.dumps/loads always produces a fresh,
# writable copy on the receiving end, exactly like the original one-task
# design already relied on for its single return value -- this applies
# the same discipline at all four boundaries now, both directions.
def _dump(value: Any) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _remote_train(series: ForecastSeries, config: KeyWorkflowConfig):
    task_started = time.time()
    report = train_key(_copy_series(series), config)
    _prepare_trained_models_for_pickling(report)
    import ray

    ctx = ray.get_runtime_context()
    return _dump(report), task_started, time.time(), ctx.get_worker_id(), ctx.get_node_id()


def _remote_evaluate(series: ForecastSeries, config: KeyWorkflowConfig, trained_payload: bytes):
    task_started = time.time()
    trained: list[TrainedModel] = pickle.loads(trained_payload)
    report = evaluate_key(_copy_series(series), config, trained)
    import ray

    ctx = ray.get_runtime_context()
    return _dump(report), task_started, time.time(), ctx.get_worker_id(), ctx.get_node_id()


def _remote_explain(
    series: ForecastSeries,
    config: KeyWorkflowConfig,
    evaluation_payload: bytes,
    trained_payload: bytes,
):
    task_started = time.time()
    evaluation: EvaluationReport = pickle.loads(evaluation_payload)
    trained: list[TrainedModel] = pickle.loads(trained_payload)
    report = explain_key(_copy_series(series), config, evaluation, trained)
    import ray

    ctx = ray.get_runtime_context()
    return _dump(report), task_started, time.time(), ctx.get_worker_id(), ctx.get_node_id()


def _remote_rank_select(
    series: ForecastSeries,
    config: KeyWorkflowConfig,
    evaluation_payload: bytes,
    explainability_payload: bytes,
):
    task_started = time.time()
    evaluation: EvaluationReport = pickle.loads(evaluation_payload)
    explainability: ExplainabilityReport = pickle.loads(explainability_payload)
    ranking, selection = rank_select_key(_copy_series(series), config, evaluation, explainability)
    # The winning model per group lives on selection.results[].fitted_model —
    # the only one of these four returns that carries a fitted estimator.
    for result in selection.results:
        if result.fitted_model is not None:
            result.fitted_model.prepare_for_pickling()
    import ray

    ctx = ray.get_runtime_context()
    return _dump((ranking, selection)), task_started, time.time(), ctx.get_worker_id(), ctx.get_node_id()


# Fallback payloads for a key with nothing from the stage before it —
# should not happen for an active key, but never crashes a submission.
_EMPTY_TRAINED = _dump([])
_EMPTY_EVALUATION = _dump(EvaluationReport())
_EMPTY_EXPLAINABILITY = _dump(ExplainabilityReport())


def _remote_tasks():
    """Decorated lazily, only where Ray is present, so this module stays
    importable in environments that do not have it."""
    import ray as _ray

    return {
        "train": _ray.remote(num_cpus=1)(_remote_train),
        "evaluate": _ray.remote(num_cpus=1)(_remote_evaluate),
        "explain": _ray.remote(num_cpus=1)(_remote_explain),
        "rank_select": _ray.remote(num_cpus=1)(_remote_rank_select),
    }


# Most tasks that were ever running at the same instant
def _peak_overlap(spans: list[tuple[float, float]]) -> int:
    events = [(start, 1) for start, _ in spans] + [(end, -1) for _, end in spans]

    events.sort(key=lambda event: (event[0], event[1]))

    running = peak = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)
    return peak


def _iso(wall_time: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(wall_time, tz=timezone.utc).isoformat(timespec="seconds")


class StagedKeyExecution:
    """Owns one run's key-parallel state across all four stages.

    Construct once per run, then call run_training() -> run_evaluation()
    -> run_explainability() -> run_rank_select() in that order — each
    depends on the previous stage's real per-key output, never on a
    result cached from a single combined task.
    """

    def __init__(
        self,
        series_collection: list[ForecastSeries],
        config: KeyWorkflowConfig,
        *,
        use_ray: bool = True,
    ) -> None:
        self._config = config
        self._run_started_wall = time.time()
        # group_id -> series, for keys still alive going into the next stage.
        self._active: dict[str, ForecastSeries] = {s.group_id: s for s in series_collection}
        # Original series order — every stage merges in this order, not
        # completion order, so a run's output is stable across executions.
        self._original_order: list[str] = [s.group_id for s in series_collection]
        self._total_keys = len(series_collection)
        self._failures: dict[str, str] = {}
        self._trained_by_key: dict[str, list[TrainedModel]] = {}
        self._evaluation_by_key: dict[str, EvaluationReport] = {}
        self._explainability_by_key: dict[str, ExplainabilityReport] = {}

        self._use_ray = use_ray and ray_available() and bool(series_collection)
        self._topology: dict[str, Any] = {}
        self._tasks: dict[str, Any] = {}
        self._config_ref: Any = None
        if self._use_ray:
            import ray

            self._topology = _start_ray() if not ray.is_initialized() else {"ray_mode": "already_initialized"}
            self._ray_cpus = ray.cluster_resources().get("CPU", 0)
            self._config_ref = ray.put(config)
            self._tasks = _remote_tasks()
        elif use_ray and not ray_available() and series_collection:
            logger.warning("Ray is not installed; running the key-parallel stages sequentially in this process.")

    @property
    def failed_keys(self) -> dict[str, str]:
        return dict(self._failures)

    # Picklable per-key state only — never the Ray refs/remote functions
    # above, which are invalid the instant this process's Ray shuts down.
    # This is what crosses a Databricks task boundary via the checkpoint.
    def snapshot(self) -> dict[str, Any]:
        return {
            "active": dict(self._active),
            "original_order": list(self._original_order),
            "failures": dict(self._failures),
            "trained_by_key": dict(self._trained_by_key),
            "evaluation_by_key": dict(self._evaluation_by_key),
            "explainability_by_key": dict(self._explainability_by_key),
            "total_keys": self._total_keys,
        }

    # Rebuild an executor in a fresh process from a prior stage's snapshot —
    # a new Databricks task cannot reuse the last task's in-memory instance,
    # since each is a separate process.
    @classmethod
    def resume(cls, config: KeyWorkflowConfig, snapshot: dict[str, Any], *, use_ray: bool = True) -> "StagedKeyExecution":
        active: dict[str, ForecastSeries] = snapshot["active"]
        original_order: list[str] = snapshot["original_order"]
        series_collection = [active[group_id] for group_id in original_order if group_id in active]

        executor = cls(series_collection, config, use_ray=use_ray)
        executor._failures = dict(snapshot.get("failures", {}))
        executor._trained_by_key = dict(snapshot.get("trained_by_key", {}))
        executor._evaluation_by_key = dict(snapshot.get("evaluation_by_key", {}))
        executor._explainability_by_key = dict(snapshot.get("explainability_by_key", {}))
        executor._total_keys = snapshot.get("total_keys", len(series_collection))
        return executor

    # ---- Stage 1: Train -------------------------------------------------

    def run_training(self, on_progress: ProgressCallback | None = None) -> tuple[TrainingReport, dict[str, Any]]:
        def submit(group_id: str):
            series = self._active[group_id]
            if self._use_ray:
                return self._tasks["train"].remote(series, self._config_ref)
            return _remote_train(series, self._config)

        by_key, telemetry = self._run_stage("train", submit, on_progress)
        for group_id, report in by_key.items():
            self._trained_by_key[group_id] = _dump(report.trained_models())

        merged = TrainingReport()
        for group_id in self._original_order:
            if group_id in by_key:
                _merge_training_report(merged, by_key[group_id])
        return merged, telemetry

    # ---- Stage 2: Evaluate -----------------------------------------------

    def run_evaluation(self, on_progress: ProgressCallback | None = None) -> tuple[EvaluationReport, dict[str, Any]]:
        def submit(group_id: str):
            series = self._active[group_id]
            trained_payload = self._trained_by_key.get(group_id, _EMPTY_TRAINED)
            if self._use_ray:
                return self._tasks["evaluate"].remote(series, self._config_ref, trained_payload)
            return _remote_evaluate(series, self._config, trained_payload)

        by_key, telemetry = self._run_stage("evaluate", submit, on_progress)
        for group_id, report in by_key.items():
            self._evaluation_by_key[group_id] = _dump(report)

        merged = EvaluationReport()
        for group_id in self._original_order:
            if group_id in by_key:
                _merge_evaluation_report(merged, by_key[group_id])
        merged.groups_evaluated = len({r.group_id for r in merged.results})
        return merged, telemetry

    # ---- Stage 3: Explain --------------------------------------------------

    def run_explainability(self, on_progress: ProgressCallback | None = None) -> tuple[ExplainabilityReport, dict[str, Any]]:
        def submit(group_id: str):
            series = self._active[group_id]
            evaluation_payload = self._evaluation_by_key.get(group_id, _EMPTY_EVALUATION)
            trained_payload = self._trained_by_key.get(group_id, _EMPTY_TRAINED)
            if self._use_ray:
                return self._tasks["explain"].remote(series, self._config_ref, evaluation_payload, trained_payload)
            return _remote_explain(series, self._config, evaluation_payload, trained_payload)

        by_key, telemetry = self._run_stage("explain", submit, on_progress)
        for group_id, report in by_key.items():
            self._explainability_by_key[group_id] = _dump(report)

        merged = ExplainabilityReport()
        for group_id in self._original_order:
            report = by_key.get(group_id)
            if report is not None:
                merged.results.extend(report.results)
                merged.duration_seconds += report.duration_seconds
        return merged, telemetry

    # ---- Stage 4: Rank & Select --------------------------------------------

    def run_rank_select(self, on_progress: ProgressCallback | None = None) -> tuple[RankingReport, ProductionSelectionReport, dict[str, Any]]:
        def submit(group_id: str):
            series = self._active[group_id]
            evaluation_payload = self._evaluation_by_key.get(group_id, _EMPTY_EVALUATION)
            explainability_payload = self._explainability_by_key.get(group_id, _EMPTY_EXPLAINABILITY)
            if self._use_ray:
                return self._tasks["rank_select"].remote(
                    series, self._config_ref, evaluation_payload, explainability_payload
                )
            return _remote_rank_select(series, self._config, evaluation_payload, explainability_payload)

        by_key, telemetry = self._run_stage("rank_select", submit, on_progress)

        ranking = RankingReport()
        selection = ProductionSelectionReport()
        # Series order, not completion order, so a run's output is stable
        # from one execution to the next.
        for group_id in self._original_order:
            pair = by_key.get(group_id)
            if pair is None:
                continue
            ranking_part, selection_part = pair
            ranking.rankings.update(ranking_part.rankings)
            ranking.duration_seconds += ranking_part.duration_seconds
            selection.results.extend(selection_part.results)
            selection.duration_seconds += selection_part.duration_seconds
        return ranking, selection, telemetry

    # ---- shared fan-out/collect/telemetry --------------------------------

    def _run_stage(
        self, stage_name: str, submit, on_progress: ProgressCallback | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        stage_wall_started = time.time()
        keys = list(self._active)
        total = len(keys)

        if total == 0:
            return {}, self._telemetry(stage_name, stage_wall_started, total=0, task_records=[])

        if self._use_ray:
            return self._run_stage_ray(stage_name, keys, submit, stage_wall_started, on_progress)
        return self._run_stage_local(stage_name, keys, submit, stage_wall_started, on_progress)

    def _run_stage_ray(self, stage_name, keys, submit, stage_wall_started, on_progress):
        import ray

        pending = {submit(group_id): group_id for group_id in keys}
        results: dict[str, Any] = {}
        task_records: list[dict[str, Any]] = []
        spans: list[tuple[float, float]] = []

        waiting = list(pending)
        while waiting:
            done, waiting = ray.wait(waiting, num_returns=1)
            group_id = pending[done[0]]
            try:
                payload, task_started, task_finished, worker_id, node_id = ray.get(done[0])
                results[group_id] = pickle.loads(payload)
                spans.append((task_started, task_finished))
                task_records.append(
                    {
                        "group_id": group_id,
                        "worker_id": worker_id,
                        "node_id": node_id,
                        "start": round(task_started - self._run_started_wall, 3),
                        "end": round(task_finished - self._run_started_wall, 3),
                        "duration_seconds": round(task_finished - task_started, 3),
                        "status": "Completed",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - collected results are already safe
                self._failures[group_id] = f"{type(exc).__name__}: {exc}"
                del self._active[group_id]
                task_records.append({"group_id": group_id, "status": "Failed", "error": str(exc)})
                logger.exception("Key %s failed in stage %s", group_id, stage_name)

            if on_progress is not None:
                on_progress(
                    stage_name,
                    self._telemetry(
                        stage_name, stage_wall_started, total=len(keys), task_records=task_records, spans=spans
                    ),
                )

        telemetry = self._telemetry(stage_name, stage_wall_started, total=len(keys), task_records=task_records, spans=spans)
        return results, telemetry

    def _run_stage_local(self, stage_name, keys, submit, stage_wall_started, on_progress):
        results: dict[str, Any] = {}
        task_records: list[dict[str, Any]] = []
        spans: list[tuple[float, float]] = []

        for group_id in keys:
            try:
                payload, task_started, task_finished, _worker_id, _node_id = submit(group_id)
                results[group_id] = pickle.loads(payload)
                spans.append((task_started, task_finished))
                task_records.append(
                    {
                        "group_id": group_id,
                        "start": round(task_started - self._run_started_wall, 3),
                        "end": round(task_finished - self._run_started_wall, 3),
                        "duration_seconds": round(task_finished - task_started, 3),
                        "status": "Completed",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - one key must not end the run
                self._failures[group_id] = f"{type(exc).__name__}: {exc}"
                del self._active[group_id]
                task_records.append({"group_id": group_id, "status": "Failed", "error": str(exc)})
                logger.exception("Key %s failed in stage %s", group_id, stage_name)

            if on_progress is not None:
                on_progress(
                    stage_name,
                    self._telemetry(
                        stage_name, stage_wall_started, total=len(keys), task_records=task_records, spans=spans
                    ),
                )

        telemetry = self._telemetry(stage_name, stage_wall_started, total=len(keys), task_records=task_records, spans=spans)
        return results, telemetry

    def _telemetry(
        self,
        stage_name: str,
        stage_wall_started: float,
        *,
        total: int,
        task_records: list[dict[str, Any]],
        spans: list[tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        spans = spans or []
        completed = sum(1 for t in task_records if t.get("status") == "Completed")
        failed = sum(1 for t in task_records if t.get("status") == "Failed")
        now = time.time()
        telemetry: dict[str, Any] = {
            "stage": stage_name,
            "executor": "ray" if self._use_ray else ("sequential" if total else "none"),
            "total_tasks": total,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "running_tasks": max(0, total - completed - failed),
            "task_durations": [t["duration_seconds"] for t in task_records if "duration_seconds" in t],
            "tasks": task_records,
            "wall_seconds": round(now - stage_wall_started, 3),
            "started_at": _iso(stage_wall_started),
            "completed_at": _iso(now) if (completed + failed) == total else None,
            "max_concurrent_tasks": _peak_overlap(spans),
        }
        if self._use_ray:
            telemetry.update(
                ray_cpus=self._ray_cpus,
                ray_nodes_used=len({t["node_id"] for t in task_records if t.get("node_id")}),
                ray_workers_used=len({t["worker_id"] for t in task_records if t.get("worker_id")}),
                **self._topology,
            )
        return telemetry


def _merge_training_report(into: TrainingReport, part: TrainingReport) -> None:
    into.results.extend(part.results)
    into.groups_trained += part.groups_trained
    into.duration_seconds += part.duration_seconds
    if not into.models_requested:
        into.models_requested = list(part.models_requested)
    if not into.models_unavailable:
        into.models_unavailable = list(part.models_unavailable)


def _merge_evaluation_report(into: EvaluationReport, part: EvaluationReport) -> None:
    into.results.extend(part.results)
    for attr in (
        "duration_seconds",
        "backtest_seconds",
        "forecast_generation_seconds",
        "validation_seconds",
        "model_fit_count",
        "backtest_windows_evaluated",
        "forecasts_reused",
        "forecasts_refit",
    ):
        setattr(into, attr, getattr(into, attr) + getattr(part, attr))
