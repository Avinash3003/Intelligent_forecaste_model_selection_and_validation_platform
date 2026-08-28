"""Key-level parallel execution must change how the run executes, not what
it answers.

The comparison tests run the same series through `_execute_sequentially` and
`_execute_with_ray` and require the merged reports to match, because every
stage inside a key already scores that key against itself alone — so
splitting keys across processes has nothing to change.

Ray tests are skipped where Ray is not installed; the merge and collection
tests are not, since those are pure and matter everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import pytest

from forecast_engine.config.drift_config import DriftValidationConfig
from forecast_engine.config.evaluation_config import EvaluationConfig
from forecast_engine.config.explainability_config import ExplainabilityConfig
from forecast_engine.config.model_config import ModelConfig
from forecast_engine.config.pipeline_config import PipelineConfig
from forecast_engine.config.ranking_config import RankingConfig
from forecast_engine.core.forecast_configuration import AggregationMethod, ForecastConfiguration
from forecast_engine.parallel.key_workflow import (
    KeyReports,
    KeyWorkflowConfig,
    merge_key_reports,
    run_key,
)
from forecast_engine.parallel.ray_executor import (
    _MATH_THREAD_VARS,
    _execute_sequentially,
    _execute_with_ray,
    _peak_overlap,
    _worker_runtime_env,
    execute_keys,
    ray_available,
)
from forecast_engine.s01_preprocessing.data_preprocessor import DataPreprocessor
from forecast_engine.s01_preprocessing.frequency_detector import FrequencyDetector
from forecast_engine.s01_preprocessing.group_generator import GroupGenerator
from forecast_engine.s01_preprocessing.series_builder import SeriesBuilder
from forecast_engine.s08_ranking.ranking_report import RankedModel, RankingReport

requires_ray = pytest.mark.skipif(not ray_available(), reason="Ray is not installed")


# A deterministic multi-key monthly frame, built without touching disk
def _dataset(keys: int = 4, months: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2019-01-01", periods=months, freq="MS")
    rows = []
    for k in range(1, keys + 1):
        values = 400 + 30 * k + np.linspace(0, 90, months) + rng.normal(0, 12, months)
        rows.extend(
            {"date": d, "store": f"S{k}", "sales": round(float(v), 2)}
            for d, v in zip(dates, values)
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def series_collection():
    config = ForecastConfiguration(
        date_column="date",
        target_column="sales",
        key_columns=("store",),
        aggregation_method=AggregationMethod.SUM,
    )
    config.validate()

    pipeline_config = PipelineConfig.default()
    frame = _dataset()
    frequency = FrequencyDetector().detect(frame["date"])
    curated, _ = DataPreprocessor(
        pipeline_config.preprocessing,
        pipeline_config.conversion,
        pipeline_config.quality,
        pipeline_config.aggregation,
    ).prepare(frame, config, frequency)

    groups = GroupGenerator(pipeline_config.grouping).generate(curated, config)
    return SeriesBuilder(pipeline_config.grouping).build(groups, config, frequency)


@pytest.fixture(scope="module")
def workflow_config():
    # Seasonal Naive alone: no tuning grid and no optional library, so the
    # whole comparison runs twice inside a unit test. It ships disabled
    # because it is the fallback rather than a candidate, so this run enables
    # it deliberately.
    default = ModelConfig.default()
    registry = tuple(
        replace(spec, enabled=True) if spec.name == "seasonal_naive" else spec
        for spec in default.registry
    )

    return KeyWorkflowConfig(
        model=replace(default, registry=registry),
        evaluation=EvaluationConfig.default(),
        explainability=ExplainabilityConfig.default(),
        ranking=RankingConfig.default(),
        drift=DriftValidationConfig.default(),
        selected_models=("seasonal_naive",),
    )


# Durations and wall-clock stamps differ between two executions of anything;
# nothing else may differ between these two.
def _without_timings(node):
    if isinstance(node, dict):
        return {
            key: _without_timings(value)
            for key, value in node.items()
            if "seconds" not in key and not key.endswith("_at") and key != "timing_breakdown"
        }
    if isinstance(node, list):
        return [_without_timings(value) for value in node]
    return node


# Plain `==` is wrong here: a model whose interval cannot be computed reports
# NaN, and IEEE-754 says NaN != NaN, so two identical reports would compare as
# different. Both sides producing NaN in the same place IS agreement.
def _equivalent(left, right):
    if isinstance(left, float) and isinstance(right, float):
        return (math.isnan(left) and math.isnan(right)) or left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_equivalent(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_equivalent(x, y) for x, y in zip(left, right))
    return left == right


def test_one_key_workflow_covers_only_its_own_key(series_collection, workflow_config):
    reports = run_key(series_collection[0], workflow_config)
    group_id = series_collection[0].group_id

    assert {r.group_id for r in reports.training.results} == {group_id}
    assert {r.group_id for r in reports.evaluation.results} == {group_id}
    assert set(reports.ranking.rankings) <= {group_id}
    assert {r.group_id for r in reports.selection.results} == {group_id}


def test_merge_keeps_every_key_and_overwrites_none(series_collection, workflow_config):
    per_key = [run_key(series, workflow_config) for series in series_collection]
    merged = merge_key_reports(per_key)

    expected = [series.group_id for series in series_collection]
    assert [r.group_id for r in merged.selection.results] == expected
    assert merged.evaluation.groups_evaluated == len(expected)
    assert merged.training.groups_trained == len(expected)
    # Every per-key record survives the merge, none replaced by a later key's.
    assert len(merged.training.results) == sum(len(r.training.results) for r in per_key)
    assert len(merged.evaluation.results) == sum(len(r.evaluation.results) for r in per_key)


def test_merge_never_collapses_two_keys_rankings():
    """`rankings` is a dict, so a merge that mishandled it would silently
    drop a key rather than fail."""
    first, second = KeyReports(), KeyReports()
    first.ranking = RankingReport(rankings={"A": [RankedModel(group_id="A", model_name="arima")]})
    second.ranking = RankingReport(rankings={"B": [RankedModel(group_id="B", model_name="prophet")]})

    merged = merge_key_reports([first, second])

    assert sorted(merged.ranking.rankings) == ["A", "B"]
    assert merged.ranking.rankings["A"][0].model_name == "arima"
    assert merged.ranking.rankings["B"][0].model_name == "prophet"


def test_sequential_execution_returns_every_key(series_collection, workflow_config):
    reports, telemetry = _execute_sequentially(series_collection, workflow_config)

    assert telemetry["keys_succeeded"] == len(series_collection)
    assert telemetry["keys_failed"] == []
    assert len(reports.selection.results) == len(series_collection)


@requires_ray
def test_ray_execution_matches_sequential_exactly(series_collection, workflow_config):
    sequential, _ = _execute_sequentially(series_collection, workflow_config)
    parallel, telemetry = _execute_with_ray(series_collection, workflow_config)

    assert telemetry["executor"] == "ray"
    assert telemetry["keys_succeeded"] == len(series_collection)

    for name in ("training", "evaluation", "explainability", "ranking", "selection"):
        assert _equivalent(
            _without_timings(getattr(sequential, name).to_dict()),
            _without_timings(getattr(parallel, name).to_dict()),
        ), f"{name} differs between sequential and Ray execution"


@requires_ray
def test_ray_preserves_key_order_regardless_of_completion_order(series_collection, workflow_config):
    """Merged in series order, not completion order — otherwise a run's
    reports would reshuffle from one execution to the next."""
    reports, _ = _execute_with_ray(series_collection, workflow_config)

    assert [r.group_id for r in reports.selection.results] == [s.group_id for s in series_collection]


@requires_ray
def test_one_failing_key_does_not_lose_the_others(series_collection, workflow_config):
    """A key whose task raises must cost only that key. The others were
    already collected, and collection is keyed by group id."""
    broken = [*series_collection]
    # A series whose observations cannot be modelled at all: `run_key`'s own
    # per-model guards do not cover a series object that fails on access.
    broken[1] = _PoisonSeries(series_collection[1].group_id)

    reports, telemetry = _execute_with_ray(broken, workflow_config)

    assert telemetry["keys_failed"] == [series_collection[1].group_id]
    assert telemetry["keys_succeeded"] == len(series_collection) - 1
    survivors = {r.group_id for r in reports.selection.results}
    assert survivors == {s.group_id for s in series_collection} - {series_collection[1].group_id}


@requires_ray
def test_ray_returns_a_usable_fitted_arima(series_collection):
    """A fitted ARIMA has to survive the trip back from the worker.

    Its state-space object is Cython and rejects the read-only buffers Ray
    produces when it shares object-store memory instead of copying it, so
    this pair of keys is the regression guard for that: a model that trains
    but cannot be read back would break Persist Models, not this stage.
    """
    config = KeyWorkflowConfig(
        model=ModelConfig.default(),
        evaluation=EvaluationConfig.default(),
        explainability=ExplainabilityConfig.default(),
        ranking=RankingConfig.default(),
        drift=DriftValidationConfig.default(),
        selected_models=("arima",),
    )

    reports, telemetry = _execute_with_ray(series_collection[:2], config)

    assert telemetry["keys_failed"] == []
    fitted = [r.fitted_model for r in reports.training.results if r.fitted_model is not None]
    assert fitted, "no ARIMA came back fitted"
    # Persist Models reads exactly this, so it has to be a live object.
    assert all(model.model is not None for model in fitted)


@requires_ray
def test_ray_schedules_against_the_cpus_it_finds(series_collection, workflow_config):
    _, telemetry = _execute_with_ray(series_collection, workflow_config)

    assert telemetry["ray_cpus"] >= 1
    # Never more keys at once than Ray had CPUs to give — each task asks for
    # exactly one, so this is the scheduler's own bound, not a fixed number.
    assert 1 <= telemetry["max_concurrent_keys"] <= telemetry["ray_cpus"]


# ---- worker thread caps ------------------------------------------------
#
# A task is charged one CPU, so it must not then start one math thread per
# core. Four keys x four OpenMP threads on a 4-vCPU node is 16 threads over
# 4 cores, and every key runs slower for it.


def test_worker_env_caps_every_math_library_to_one_thread(monkeypatch):
    for name in _MATH_THREAD_VARS:
        monkeypatch.delenv(name, raising=False)

    env = _worker_runtime_env()["env_vars"]

    for name in _MATH_THREAD_VARS:
        assert env[name] == "1", name


def test_worker_env_covers_openmp_and_the_blas_backends():
    """Pinning only one of these leaves the others free to oversubscribe."""
    assert set(_MATH_THREAD_VARS) == {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }


def test_a_driver_thread_count_does_not_leak_into_the_workers(monkeypatch):
    """The driver holds the whole node; a worker holds one CPU.

    Ray defaults a worker to its CPU share only when the variable is unset --
    when the driver has one, Ray passes it straight through. So on a platform
    that exports OMP_NUM_THREADS=<node cores>, inheriting it would give every
    task one thread per core and oversubscribe the node by the number of
    concurrent keys. The cap is deliberately not inherited.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")

    env = _worker_runtime_env()["env_vars"]

    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"


def test_capping_threads_does_not_disturb_the_pythonpath_entry(monkeypatch):
    """Both concerns share one env dict; neither may displace the other."""
    monkeypatch.delenv("PYTHONPATH", raising=False)

    env = _worker_runtime_env()["env_vars"]

    assert env["PYTHONPATH"].endswith("tech_demo") or "tech_demo" in env["PYTHONPATH"]
    assert env["OMP_NUM_THREADS"] == "1"


@pytest.mark.skipif(not ray_available(), reason="Ray is not installed")
def test_the_caps_actually_reach_a_running_ray_worker():
    """The contract that matters is the one inside the worker process, not
    the dict the driver built: Ray must apply these before the worker
    imports numpy, which is when the libraries read them.

    Ray sets OMP_NUM_THREADS to a task's CPU share by itself, so three of
    these four would hold even without the runtime env. NUMEXPR_NUM_THREADS
    is the one it does not set, and numexpr reads os.cpu_count() instead of
    OMP_NUM_THREADS -- so this asserts all four rather than trusting that
    Ray's default keeps covering the other three."""
    import ray

    if not ray.is_initialized():
        from forecast_engine.parallel.ray_executor import _start_ray

        _start_ray()

    @ray.remote(num_cpus=1)
    def _read_env():
        import os

        return {name: os.environ.get(name) for name in _MATH_THREAD_VARS}

    seen = ray.get(_read_env.remote())

    assert seen == {name: "1" for name in _MATH_THREAD_VARS}


def test_execute_keys_handles_an_empty_run():
    reports, telemetry = execute_keys([], _EMPTY_CONFIG)

    assert telemetry["keys"] == 0
    assert reports.selection.results == []


@pytest.mark.parametrize(
    "spans, expected",
    [
        ([], 0),
        ([(0.0, 1.0)], 1),
        ([(0.0, 1.0), (2.0, 3.0)], 1),
        ([(0.0, 2.0), (1.0, 3.0)], 2),
        ([(0.0, 4.0), (1.0, 2.0), (1.5, 3.0)], 3),
        # Touching, not overlapping: one ends exactly as the next begins.
        ([(0.0, 1.0), (1.0, 2.0)], 1),
    ],
)
def test_peak_overlap_counts_simultaneous_tasks(spans, expected):
    assert _peak_overlap(spans) == expected


_EMPTY_CONFIG = KeyWorkflowConfig(
    model=ModelConfig.default(),
    evaluation=EvaluationConfig.default(),
    explainability=ExplainabilityConfig.default(),
    ranking=RankingConfig.default(),
    drift=DriftValidationConfig.default(),
)


class _PoisonFrame:
    """A frame that dies the moment its task touches it."""

    def copy(self):
        raise RuntimeError("this key cannot be processed")


@dataclass
class _PoisonSeries:
    """Stands in for a key whose workflow dies outright inside its task.

    A dataclass because the Ray task rebuilds its series with
    `dataclasses.replace`, so this has to fail the way a real key would —
    inside the task, not before it starts.
    """

    group_id: str
    frame: _PoisonFrame = field(default_factory=_PoisonFrame)
    key_values: dict[str, str] = field(default_factory=dict)
