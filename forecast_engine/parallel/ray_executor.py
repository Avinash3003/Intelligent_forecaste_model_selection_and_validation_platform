"""Runs each forecast key's complete workflow as an independent Ray task.

The unit of parallelism is the KEY, never the model: one task owns one key's
whole train -> evaluate -> explain -> rank -> select sequence, so nothing
inside a key ever crosses a process boundary.

Each task asks for one CPU and Ray decides how many run at once from the
resources it actually finds, so a queued key starts the moment a running one
frees its slot. Nothing here assumes a particular core count.

Falls back to running the same `run_key` calls in-process when Ray is not
installed, which keeps this importable everywhere and makes the sequential
and parallel paths provably the same code.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from forecast_engine.parallel.key_workflow import (
    KeyReports,
    KeyWorkflowConfig,
    merge_key_reports,
    run_key,
)
from forecast_engine.s01_preprocessing.series_builder import ForecastSeries

logger = logging.getLogger(__name__)


# True when Ray can be imported in this environment
def ray_available() -> bool:
    try:
        import ray  # noqa: F401
    except ImportError:
        return False
    return True


def execute_keys(
    series_collection: list[ForecastSeries],
    config: KeyWorkflowConfig,
    *,
    use_ray: bool = True,
) -> tuple[KeyReports, dict[str, Any]]:
    """Run every key's workflow and merge the results.

    Returns the merged reports plus telemetry describing how the run was
    actually executed — the backend records it on the run summary, so a
    finished run says how many keys ran in parallel rather than only that
    parallelism was requested.
    """
    if not series_collection:
        return KeyReports(), {"executor": "none", "keys": 0}

    if use_ray and ray_available():
        return _execute_with_ray(series_collection, config)

    if use_ray:
        logger.warning("Ray is not installed; running keys sequentially in this process.")
    return _execute_sequentially(series_collection, config)


# One key at a time, in this process — the reference path
def _execute_sequentially(
    series_collection: list[ForecastSeries], config: KeyWorkflowConfig
) -> tuple[KeyReports, dict[str, Any]]:
    started = time.perf_counter()
    per_key, failures = [], {}

    for series in series_collection:
        try:
            per_key.append(run_key(series, config))
        except Exception as exc:  # noqa: BLE001 - one key must not end the run
            failures[series.group_id] = f"{type(exc).__name__}: {exc}"
            logger.exception("Key %s failed", series.group_id)

    return merge_key_reports(per_key), {
        "executor": "sequential",
        "keys": len(series_collection),
        "keys_succeeded": len(per_key),
        "keys_failed": sorted(failures),
        "failures": failures,
        "max_concurrent_keys": 1 if per_key else 0,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }


# Every key as its own Ray task, collected as each one finishes
def _execute_with_ray(
    series_collection: list[ForecastSeries], config: KeyWorkflowConfig
) -> tuple[KeyReports, dict[str, Any]]:
    import ray

    # Ray may already be running (a second call in one process); there is
    # then no topology decision to report for this call.
    topology = _start_ray() if not ray.is_initialized() else {"ray_mode": "already_initialized"}

    started = time.perf_counter()
    # A wall-clock reference alongside the monotonic one above:
    # `task_started`/`task_finished` below are wall-clock (time.time()),
    # since perf_counter has no fixed epoch and cannot be compared across
    # processes — every Ray worker is its own process. This is the anchor
    # every key_span's offset is measured from.
    run_started_wall = time.time()
    cpus = ray.cluster_resources().get("CPU", 0)

    # Shared once rather than serialized into every task.
    config_ref = ray.put(config)
    pending = {_run_key_task.remote(series, config_ref): series.group_id for series in series_collection}

    # Keyed by group id, so reusing a worker for the next key can never
    # overwrite the result of the key it finished.
    by_group: dict[str, KeyReports] = {}
    failures: dict[str, str] = {}
    spans: list[tuple[float, float]] = []
    # One entry per key that actually completed — a failed key has no
    # result tuple to read a span from, so it is simply absent here rather
    # than reported with a fabricated timing.
    key_spans: list[dict[str, Any]] = []

    waiting = list(pending)
    while waiting:
        done, waiting = ray.wait(waiting, num_returns=1)
        group_id = pending[done[0]]
        try:
            payload, task_started, task_finished, worker_id, node_id = ray.get(done[0])
            by_group[group_id] = pickle.loads(payload)
            spans.append((task_started, task_finished))
            key_spans.append(
                {
                    "group_id": group_id,
                    "worker_id": worker_id,
                    "node_id": node_id,
                    "start": round(task_started - run_started_wall, 3),
                    "end": round(task_finished - run_started_wall, 3),
                }
            )
        except Exception as exc:  # noqa: BLE001 - collected results are already safe
            failures[group_id] = f"{type(exc).__name__}: {exc}"
            logger.exception("Key %s failed in its Ray task", group_id)

    # Merged in series order, not completion order, so the reports are
    # byte-identical to a sequential run's.
    ordered = [by_group[s.group_id] for s in series_collection if s.group_id in by_group]

    return merge_key_reports(ordered), {
        "executor": "ray",
        "ray_version": ray.__version__,
        "ray_cpus": cpus,
        "keys": len(series_collection),
        "keys_succeeded": len(ordered),
        "keys_failed": sorted(failures),
        "failures": failures,
        "max_concurrent_keys": _peak_overlap(spans),
        "wall_seconds": round(time.perf_counter() - started, 3),
        # How Ray was actually started, and how far the work really spread.
        # `ray_nodes_used` is counted from the spans themselves — the number
        # of distinct machines that genuinely ran a key — so multi-node
        # execution can never be inferred from the cluster's shape alone. A
        # 4-node cluster whose keys all landed on the driver reports 1 here,
        # which is the whole point.
        **topology,
        "ray_nodes_used": len({s["node_id"] for s in key_spans}),
        "ray_workers_used": len({s["worker_id"] for s in key_spans}),
        # Multi-node caveat: on Ray-on-Spark, each key_span's start/end is
        # wall-clock time on whichever node ran that key, offset against the
        # DRIVER's run_started_wall — accurate to node-to-node clock skew
        # (sub-second under NTP; exact on the single-node path, where every
        # worker shares the driver's clock).
        "key_spans": key_spans,
    }


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
    resources either way, and each key still asks for exactly one CPU.

    Returns how Ray was actually started, so the run can REPORT it rather
    than leave it to be inferred. Falling back to the driver is a legitimate
    outcome on single-node compute and a silent, expensive defect on
    multi-node compute — a cluster whose workers were paid for and never
    used. The two are indistinguishable from the outside unless the
    decision is recorded, so this returns it and `_execute_with_ray` puts it
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


# The native math libraries every model sits on, and the variable each one
# reads to decide how many threads to start.
#
# A task holds exactly one CPU (the `_run_key_task` decorator below), but OpenMP and the BLAS
# implementations default to one thread PER CORE, not per allotted CPU. On
# the 4-vCPU node this runs on, four concurrent keys would therefore start
# four threads each — 16 threads contending for 4 cores, with every task
# slowed by the other three. Capping them to one keeps a task's real CPU
# usage equal to the CPU Ray charged it for.
#
# XGBoost, LightGBM and the tuner are already pinned to `n_jobs=1` in the
# model configuration; these cover what that cannot reach — the BLAS/LAPACK
# calls underneath statsmodels (ARIMA) and numpy.
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
    # Pinned to 1 rather than inherited from the driver. Ray only defaults a
    # worker to its CPU share when the variable is UNSET; when the driver has
    # one set, Ray passes that value straight through — so on a platform that
    # exports OMP_NUM_THREADS=<node cores>, every task would start one thread
    # per core and four concurrent keys would oversubscribe 4:1. The driver's
    # value describes the driver, which holds the whole node; it is not a
    # statement about a worker that Ray charged for exactly one CPU.
    env_vars.update({name: "1" for name in _MATH_THREAD_VARS})
    return {"env_vars": env_vars}


# Asks for one CPU, so Ray's own scheduler decides the concurrency
def _remote_run_key(
    series: ForecastSeries, config: KeyWorkflowConfig
) -> tuple[bytes, float, float, str, str]:
    task_started = time.time()
    # Ray shares its object store's memory rather than copying it, so arrays
    # arrive read-only in both directions and the Cython in statsmodels
    # refuses them. Copying the frame fixes the way in; returning an opaque
    # bytes payload fixes the way out, since bytes give Ray no buffers to
    # share and the driver unpickles ordinary writable arrays. Without the
    # second half, a fitted ARIMA cannot be read back at all.
    series = replace(series, frame=series.frame.copy())
    reports = run_key(series, config)
    # Which worker/node actually ran this key — the one thing that turns a
    # start/end timestamp into proof of parallelism rather than just
    # duration. Read from inside the task: it is only meaningful here,
    # never on the driver. Imported locally, matching every other Ray
    # import in this module — this function must stay importable (as a
    # plain function, never called) where Ray is not installed.
    import ray

    context = ray.get_runtime_context()
    worker_id = context.get_worker_id()
    node_id = context.get_node_id()
    return (
        pickle.dumps(reports, protocol=pickle.HIGHEST_PROTOCOL),
        task_started,
        time.time(),
        worker_id,
        node_id,
    )


# Most tasks that were ever running at the same instant
def _peak_overlap(spans: list[tuple[float, float]]) -> int:
    events = [(start, 1) for start, _ in spans] + [(end, -1) for _, end in spans]
    # Ends before starts at equal timestamps, so touching spans do not count
    # as overlapping.
    events.sort(key=lambda event: (event[0], event[1]))

    running = peak = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)
    return peak


# Decorated at import only when Ray is present, so this module stays
# importable in environments that do not have it.
if ray_available():
    import ray as _ray

    _run_key_task = _ray.remote(num_cpus=1)(_remote_run_key)
else:  # pragma: no cover - exercised only where Ray is absent
    _run_key_task = None
