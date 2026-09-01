"""Key-level parallel execution must change how the run executes, not what
it answers.

The comparison tests run the same series through StagedKeyExecution with
use_ray=False and use_ray=True and require the merged reports to match,
because every stage inside a key already scores that key against itself
alone — so splitting keys across processes has nothing to change.

Four real stage boundaries, not one: Train, Evaluate, Explain and
Rank & Select are each their own Ray fan-out across every key, run in that
order, each depending only on the stage before it.

Ray tests are skipped where Ray is not installed; the merge and collection
tests are not, since those are pure and matter everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

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
    StagedKeyExecution,
    _peak_overlap,
    _worker_runtime_env,
    ray_available,
)
from forecast_engine.s01_preprocessing.data_preprocessor import DataPreprocessor
from forecast_engine.s01_preprocessing.frequency_detector import FrequencyDetector
from forecast_engine.s01_preprocessing.group_generator import GroupGenerator
from forecast_engine.s01_preprocessing.series_builder import SeriesBuilder
from forecast_engine.s08_ranking.ranking_report import RankedModel, RankingReport

requires_ray = pytest.mark.skipif(not ray_available(), reason="Ray is not installed")


# Runs all four real stages through one executor, in order, returning
# merged reports (KeyReports shape) plus each stage's own telemetry.
def _run_all_stages(executor: StagedKeyExecution) -> tuple[KeyReports, dict[str, dict[str, Any]]]:
    training, t_train = executor.run_training()
    evaluation, t_evaluate = executor.run_evaluation()
    explainability, t_explain = executor.run_explainability()
    ranking, selection, t_rank_select = executor.run_rank_select()
    return (
        KeyReports(training, evaluation, explainability, ranking, selection),
        {"train": t_train, "evaluate": t_evaluate, "explain": t_explain, "rank_select": t_rank_select},
    )


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
    executor = StagedKeyExecution(series_collection, workflow_config, use_ray=False)
    reports, telemetry = _run_all_stages(executor)

    assert executor.failed_keys == {}
    assert len(reports.selection.results) == len(series_collection)
    for stage in ("train", "evaluate", "explain", "rank_select"):
        assert telemetry[stage]["completed_tasks"] == len(series_collection)
        assert telemetry[stage]["failed_tasks"] == 0


@requires_ray
def test_ray_execution_matches_sequential_exactly(series_collection, workflow_config):
    sequential, _ = _run_all_stages(StagedKeyExecution(series_collection, workflow_config, use_ray=False))
    parallel, telemetry = _run_all_stages(StagedKeyExecution(series_collection, workflow_config, use_ray=True))

    assert telemetry["train"]["executor"] == "ray"
    assert telemetry["train"]["completed_tasks"] == len(series_collection)

    for name in ("training", "evaluation", "explainability", "ranking", "selection"):
        assert _equivalent(
            _without_timings(getattr(sequential, name).to_dict()),
            _without_timings(getattr(parallel, name).to_dict()),
        ), f"{name} differs between sequential and Ray staged execution"


@requires_ray
def test_every_stage_is_its_own_genuine_ray_fan_out(series_collection, workflow_config):
    """The core of the refactor: four separate Ray fan-outs, not one task
    doing a key's whole workflow while three later 'stages' just relabel
    its already-finished output."""
    executor = StagedKeyExecution(series_collection, workflow_config, use_ray=True)
    _, telemetry = _run_all_stages(executor)

    for stage in ("train", "evaluate", "explain", "rank_select"):
        assert telemetry[stage]["executor"] == "ray"
        assert telemetry[stage]["total_tasks"] == len(series_collection)
        assert telemetry[stage]["completed_tasks"] == len(series_collection)
        # Each stage's own tasks, timed at its own boundary — not another
        # stage's numbers copied over.
        assert len(telemetry[stage]["tasks"]) == len(series_collection)
        assert telemetry[stage]["wall_seconds"] > 0.0


@requires_ray
def test_ray_preserves_key_order_regardless_of_completion_order(series_collection, workflow_config):
    """Merged in series order, not completion order — otherwise a run's
    reports would reshuffle from one execution to the next."""
    executor = StagedKeyExecution(series_collection, workflow_config, use_ray=True)
    reports, _ = _run_all_stages(executor)

    assert [r.group_id for r in reports.selection.results] == [s.group_id for s in series_collection]


@requires_ray
def test_one_failing_key_does_not_lose_the_others(series_collection, workflow_config):
    """A key whose task raises must cost only that key, at whichever stage
    it fails, and take no further part in any stage after that."""
    broken = [*series_collection]
    # A series whose observations cannot be modelled at all: the workflow's
    # own per-model guards do not cover a series object that fails on access.
    broken[1] = _PoisonSeries(series_collection[1].group_id)

    executor = StagedKeyExecution(broken, workflow_config, use_ray=True)
    reports, telemetry = _run_all_stages(executor)

    failed_group = series_collection[1].group_id
    assert set(executor.failed_keys) == {failed_group}
    assert telemetry["train"]["failed_tasks"] == 1
    # Dropped after failing Train -- no later stage even attempts it.
    for stage in ("evaluate", "explain", "rank_select"):
        assert telemetry[stage]["total_tasks"] == len(series_collection) - 1
        assert failed_group not in {t["group_id"] for t in telemetry[stage]["tasks"]}
    survivors = {r.group_id for r in reports.selection.results}
    assert survivors == {s.group_id for s in series_collection} - {failed_group}


@requires_ray
def test_task_records_carry_a_resolvable_worker_and_node_id(series_collection, workflow_config):
    """worker_id/node_id are what turns a timestamp into proof of
    parallelism -- without them two overlapping tasks could just as well be
    the same worker, measured wrong."""
    executor = StagedKeyExecution(series_collection, workflow_config, use_ray=True)
    _, t_train = executor.run_training()

    for task in t_train["tasks"]:
        assert task["worker_id"]
        assert task["node_id"]
        assert isinstance(task["worker_id"], str)
        assert isinstance(task["node_id"], str)


@requires_ray
def test_task_offsets_are_from_run_start_not_raw_timestamps(series_collection, workflow_config):
    """A UI renders these as a Gantt chart from x=0, not as epoch time."""
    executor = StagedKeyExecution(series_collection, workflow_config, use_ray=True)
    _, t_train = executor.run_training()

    for task in t_train["tasks"]:
        assert task["start"] >= 0.0
        assert task["end"] >= task["start"]
        # Well inside a single test run's real wall-clock budget -- catches
        # an accidental raw time.time() (a ~1.7-billion-second epoch value)
        # slipping back in.
        assert task["end"] < 300.0


@requires_ray
def test_a_failed_task_has_no_fabricated_duration(series_collection, workflow_config):
    """A key with no result tuple to read a timing from must report Failed
    with no duration, never a made-up start/end."""
    broken = [*series_collection]
    broken[1] = _PoisonSeries(series_collection[1].group_id)

    executor = StagedKeyExecution(broken, workflow_config, use_ray=True)
    _, t_train = executor.run_training()

    failed_group = series_collection[1].group_id
    failed_task = next(t for t in t_train["tasks"] if t["group_id"] == failed_group)
    assert failed_task["status"] == "Failed"
    assert "duration_seconds" not in failed_task


def test_sequential_stage_telemetry_reports_no_worker_or_node_ids(series_collection, workflow_config):
    """worker/node distribution is a Ray-only concept -- the sequential
    fallback has none, and must not fabricate one."""
    executor = StagedKeyExecution(series_collection, workflow_config, use_ray=False)
    _, t_train = executor.run_training()

    assert t_train["executor"] == "sequential"
    for task in t_train["tasks"]:
        assert "worker_id" not in task


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

    executor = StagedKeyExecution(series_collection[:2], config, use_ray=True)
    training, _ = executor.run_training()

    assert executor.failed_keys == {}
    fitted = [r.fitted_model for r in training.results if r.fitted_model is not None]
    assert fitted, "no ARIMA came back fitted"
    # Persist Models reads exactly this, so it has to be a live object.
    assert all(model.model is not None for model in fitted)


@requires_ray
def test_ray_schedules_against_the_cpus_it_finds(series_collection, workflow_config):
    executor = StagedKeyExecution(series_collection, workflow_config, use_ray=True)
    _, t_train = executor.run_training()

    assert t_train["ray_cpus"] >= 1
    # Never more keys at once than Ray had CPUs to give — each task asks for
    # exactly one, so this is the scheduler's own bound, not a fixed number.
    assert 1 <= t_train["max_concurrent_tasks"] <= t_train["ray_cpus"]


@requires_ray
def test_on_progress_fires_once_per_completed_task(series_collection, workflow_config):
    """The live-progress contract: a caller sees tasks complete one by
    one, not just the final count once the whole stage is done."""
    calls = []
    executor = StagedKeyExecution(series_collection, workflow_config, use_ray=True)

    executor.run_training(on_progress=lambda stage, telemetry: calls.append((stage, telemetry["completed_tasks"])))

    assert len(calls) == len(series_collection)
    assert all(stage == "train" for stage, _ in calls)
    assert [count for _, count in calls] == list(range(1, len(series_collection) + 1))


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

    # The engine's own parent directory -- not a specific checkout-folder
    # name, which differs between a developer's clone and a CI runner.
    engine_parent = str(Path(__file__).resolve().parents[2])
    assert env["PYTHONPATH"] == engine_parent
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
    Ray's default keeps covering the other three.

    Restarts Ray unconditionally rather than reusing whatever is already
    running -- runtime_env only ever applies at the FIRST ray.init() in a
    process, so a suite where some other test started Ray first would
    silently check that test's env, not this one's own _start_ray() call.
    """
    import ray

    from forecast_engine.parallel.ray_executor import _start_ray

    if ray.is_initialized():
        ray.shutdown()
    _start_ray()

    @ray.remote(num_cpus=1)
    def _read_env():
        import os

        return {name: os.environ.get(name) for name in _MATH_THREAD_VARS}

    seen = ray.get(_read_env.remote())

    assert seen == {name: "1" for name in _MATH_THREAD_VARS}


def test_staged_execution_handles_an_empty_run():
    executor = StagedKeyExecution([], _EMPTY_CONFIG, use_ray=True)
    reports, telemetry = _run_all_stages(executor)

    assert telemetry["train"]["total_tasks"] == 0
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
