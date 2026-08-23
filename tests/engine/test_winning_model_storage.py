"""Durable persistence of each forecast key's winning fitted model.

Before this existed the only model persistence was MLflow's, which wrapped
a *frozen forecast* rather than the fitted estimator — so nothing a run
produced could be reloaded and re-served. These cover the writer's contract:
winners only, one file per key, safe filenames, and a model that survives
the process that made it.
"""

import pickle
from dataclasses import dataclass, field
from typing import Any

import pytest

from forecast_engine.config.pipeline_config import ModelStorageConfig
from forecast_engine.s03_storage.model_writer import WinningModelWriter, sanitize_forecast_key


class _Estimator:
    """Stands in for a fitted model: carries state and predicts from it."""

    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.fit_calls = 0

    def train(self, series):  # pragma: no cover - must never be called here
        self.fit_calls += 1

    def predict(self, horizon: int) -> list[float]:
        return [self.offset + i for i in range(horizon)]


@dataclass
class _Trained:
    group_id: str
    model_name: str
    fitted_model: Any = None


@dataclass
class _Winner:
    group_id: str | None
    model_name: str | None
    status: Any = None
    fitted_model: Any = None
    key_values: dict = field(default_factory=dict)


def _writer(tmp_path):
    return WinningModelWriter(ModelStorageConfig(root_dir=str(tmp_path / "models")))


# ---------------------------------------------------------------------
# Which model gets persisted
# ---------------------------------------------------------------------


def test_only_the_winning_model_is_persisted(tmp_path):
    """A run trains keys x candidates; only the winner is durable."""
    trained = [
        _Trained("k1", "prophet", _Estimator(1)),
        _Trained("k1", "lightgbm", _Estimator(2)),  # a losing candidate
        _Trained("k1", "arima", _Estimator(3)),  # another loser
    ]
    winners = [_Winner("k1", "prophet")]

    results = _writer(tmp_path).write_all(winners, trained, "run-1")

    assert len(results) == 1
    assert results[0]["persisted"] is True
    assert results[0]["model_name"] == "prophet"
    # Exactly one file, for exactly the winner.
    assert len(list((tmp_path / "models" / "run-1").glob("*.pkl"))) == 1


def test_the_persisted_model_is_the_one_that_won(tmp_path):
    # The winner is identified by (group, model), not by training order.
    trained = [_Trained("k1", "lightgbm", _Estimator(99)), _Trained("k1", "prophet", _Estimator(7))]
    results = _writer(tmp_path).write_all([_Winner("k1", "prophet")], trained, "run-1")

    loaded = pickle.load(open(results[0]["uri"], "rb"))
    assert loaded.offset == 7, "the losing candidate's estimator was saved instead"


def test_a_fallback_winner_uses_the_estimator_selection_already_fitted(tmp_path):
    """A fallback is fitted during selection, not training, so it is not in
    the training records — it travels on the result itself."""
    winner = _Winner("k1", "seasonal_naive", fitted_model=_Estimator(5))
    results = _writer(tmp_path).write_all([winner], [], "run-1")

    assert results[0]["persisted"] is True
    assert pickle.load(open(results[0]["uri"], "rb")).offset == 5


def test_a_group_with_no_production_model_is_reported_not_written(tmp_path):
    results = _writer(tmp_path).write_all([_Winner("k1", None)], [], "run-1")

    assert results[0]["persisted"] is False
    assert "No production model" in results[0]["error"]
    assert not (tmp_path / "models" / "run-1").exists()


def test_no_retraining_happens_to_serialize_a_model(tmp_path):
    """The fitted estimator is reused; persistence must never refit."""
    estimator = _Estimator(1)
    _writer(tmp_path).write_all([_Winner("k1", "prophet")], [_Trained("k1", "prophet", estimator)], "r")

    assert estimator.fit_calls == 0


# ---------------------------------------------------------------------
# Paths and filenames
# ---------------------------------------------------------------------


def test_path_contains_the_run_id_and_the_key(tmp_path):
    results = _writer(tmp_path).write_all(
        [_Winner("1 | 4", "prophet")], [_Trained("1 | 4", "prophet", _Estimator(1))], "dbx-run-abc"
    )
    uri = results[0]["uri"]
    assert "dbx-run-abc" in uri
    assert uri.endswith("1_4_model.pkl")


def test_different_keys_cannot_collide(tmp_path):
    trained = [_Trained("1 | 1", "p", _Estimator(1)), _Trained("1 | 2", "p", _Estimator(2))]
    winners = [_Winner("1 | 1", "p"), _Winner("1 | 2", "p")]

    results = _writer(tmp_path).write_all(winners, trained, "run-1")

    uris = {r["uri"] for r in results}
    assert len(uris) == 2
    assert len(list((tmp_path / "models" / "run-1").glob("*.pkl"))) == 2


def test_different_runs_cannot_overwrite_each_other(tmp_path):
    trained = [_Trained("k1", "p", _Estimator(1))]
    a = _writer(tmp_path).write_all([_Winner("k1", "p")], trained, "run-a")
    b = _writer(tmp_path).write_all([_Winner("k1", "p")], trained, "run-b")

    assert a[0]["uri"] != b[0]["uri"]
    assert (tmp_path / "models" / "run-a").exists()
    assert (tmp_path / "models" / "run-b").exists()


@pytest.mark.parametrize(
    "raw",
    ["../../etc/passwd", "/absolute/path", "a/b/c", "..", ".", "  ", "key with spaces"],
)
def test_keys_cannot_escape_the_run_directory(tmp_path, raw):
    """Traversal is prevented structurally: separators stop being separators."""
    safe = sanitize_forecast_key(raw)
    assert "/" not in safe and ".." not in safe
    assert not safe.startswith(("/", "."))

    results = _writer(tmp_path).write_all(
        [_Winner(raw, "p")], [_Trained(raw, "p", _Estimator(1))], "run-1"
    )
    written = results[0]["uri"]
    # Whatever the key contained, the file lands inside this run's directory.
    assert written.startswith(str(tmp_path / "models" / "run-1"))


def test_a_malicious_run_id_also_cannot_escape(tmp_path):
    results = _writer(tmp_path).write_all(
        [_Winner("k", "p")], [_Trained("k", "p", _Estimator(1))], "../../escape"
    )
    assert results[0]["uri"].startswith(str(tmp_path / "models"))
    assert ".." not in results[0]["uri"]


# ---------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------


def test_the_model_reloads_and_predicts_identically(tmp_path):
    """The point of persisting it: the same forecast, from a new process."""
    estimator = _Estimator(3.5)
    results = _writer(tmp_path).write_all(
        [_Winner("k1", "prophet")], [_Trained("k1", "prophet", estimator)], "run-1"
    )

    reloaded = pickle.load(open(results[0]["uri"], "rb"))
    assert reloaded is not estimator
    assert reloaded.predict(6) == estimator.predict(6)


def test_disabled_storage_writes_nothing(tmp_path):
    writer = WinningModelWriter(ModelStorageConfig(enabled=False, root_dir=str(tmp_path / "models")))
    assert writer.write_all([_Winner("k", "p")], [_Trained("k", "p", _Estimator(1))], "r") == []
    assert not (tmp_path / "models").exists()
