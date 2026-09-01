"""Every heuristic constant estimation_service.py mirrors from the engine
must actually still match it.

These are correctly architected -- named after what they mirror, commented
with exactly which engine file and class they come from -- but nothing
verified that architecture held. The horizon bounds (see
test_run_limits_match_the_engine.py) were a real instance of this same
shape drifting into three independent, disconnected copies; these
constants are the same risk, just not yet caught out.

The engine cannot be imported from this process (separate package, its
own venv), so its source is read as text and the values compared -- the
same technique test_stage_trail.py and test_run_limits_match_the_engine.py
already use for the same reason.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.services.estimation_service import (
    _BACKTEST_HORIZON,
    _BACKTEST_MAX_WINDOWS,
    _BACKTEST_MIN_TRAIN_SIZE,
    _MODEL_MIN_OBSERVATIONS,
    _TUNING_MIN_OBSERVATIONS,
)

_ENGINE_ROOT = Path(__file__).resolve().parents[2] / "forecast_engine"
_EVALUATION_CONFIG = (_ENGINE_ROOT / "config" / "evaluation_config.py").read_text()
_MODEL_CONFIG = (_ENGINE_ROOT / "config" / "model_config.py").read_text()


def _int_field(source: str, name: str) -> int:
    match = re.search(rf"\b{name}: int = (\d+)", source)
    assert match, f"{name} not found — has evaluation_config.py's BacktestConfig changed shape?"
    return int(match.group(1))


def test_backtest_defaults_match_evaluation_configs_backtestconfig():
    assert _BACKTEST_MIN_TRAIN_SIZE == _int_field(_EVALUATION_CONFIG, "min_train_size")
    assert _BACKTEST_HORIZON == _int_field(_EVALUATION_CONFIG, "horizon")
    assert _BACKTEST_MAX_WINDOWS == _int_field(_EVALUATION_CONFIG, "max_windows")


def test_tuning_minimum_matches_tuningconfig():
    match = re.search(r"min_observations_for_tuning: int = (\d+)", _MODEL_CONFIG)
    assert match, "min_observations_for_tuning not found in model_config.py"
    assert _TUNING_MIN_OBSERVATIONS == int(match.group(1))


def test_every_models_min_observations_matches_its_modelspec():
    """Parses (name=..., ... min_observations=...) pairs in the order they
    appear -- the same shape ModelConfig.registry defines them in -- so a
    model added, removed or renumbered in the engine is caught here rather
    than silently mis-estimated."""
    pairs = re.findall(r'name="([a-z_]+)"[\s\S]*?min_observations=(\d+)', _MODEL_CONFIG)
    assert pairs, "no (name, min_observations) pairs found — has ModelSpec's shape changed?"

    engine_values = {name: int(value) for name, value in pairs}

    assert _MODEL_MIN_OBSERVATIONS == engine_values


def test_every_candidate_model_has_a_mirrored_minimum():
    """A model present in the engine's registry but missing here would
    silently fall through to _DEFAULT_MIN_OBSERVATIONS instead of its real
    bar, which is exactly the transposition bug this dict's own comment
    already records having happened once."""
    from app.config.model_availability import CANDIDATE_MODEL_IDS

    assert set(CANDIDATE_MODEL_IDS) <= set(_MODEL_MIN_OBSERVATIONS)


def test_the_default_fallback_model_matches_modelconfigs_own_default():
    from app.config.model_availability import DEFAULT_FALLBACK_MODEL

    match = re.search(r'DEFAULT_FALLBACK_MODEL = "([a-z_]+)"', _MODEL_CONFIG)
    assert match, "DEFAULT_FALLBACK_MODEL not found in model_config.py"
    assert DEFAULT_FALLBACK_MODEL == match.group(1)
