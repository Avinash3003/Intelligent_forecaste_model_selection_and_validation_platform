"""The backend's horizon bounds must be exactly what the engine enforces.

Found duplicated three times with no shared source: `run_pipeline.py`
defined MIN_FORECAST_HORIZON=6 / MAX_FORECAST_HORIZON=60 on its own, and
`deployment.py` / `estimation.py` each separately wrote `ge=6, le=60` as
bare literals. A request the backend accepted but the engine then rejected
-- or the reverse, an engine capability the backend silently refused --
would have been a real validation gap, not a cosmetic inconsistency, and
nothing would have caught the two drifting apart.

Consolidated behind one backend-side constant (app.config.run_limits), and
verified here against the engine's own source text, the same way
test_stage_trail.py verifies PIPELINE_STAGES against `begin_stage(...)`
calls -- the engine cannot be imported from this process (a separate
package, its own venv), so the source is read as text rather than trusted
to match by convention.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config.run_limits import DEFAULT_FORECAST_HORIZON, MAX_FORECAST_HORIZON, MIN_FORECAST_HORIZON
from app.schemas.deployment import DeploymentRequest
from app.schemas.estimation import EstimationRequest
from app.schemas.metadata import MetadataMapping

_ENGINE_SOURCE = (
    Path(__file__).resolve().parents[2] / "forecast_engine" / "run_pipeline.py"
).read_text()


def _engine_constant(name: str) -> int:
    match = re.search(rf"^{name} = (\d+)", _ENGINE_SOURCE, re.MULTILINE)
    assert match, f"{name} not found in run_pipeline.py — has it been renamed?"
    return int(match.group(1))


def test_the_backend_bounds_are_read_from_the_engines_own_source():
    assert MIN_FORECAST_HORIZON == _engine_constant("MIN_FORECAST_HORIZON")
    assert MAX_FORECAST_HORIZON == _engine_constant("MAX_FORECAST_HORIZON")


def _metadata():
    return MetadataMapping(date_column="date", target_column="sales", key_columns=["store"], feature_columns=[])


def test_deployment_request_uses_the_shared_bounds():
    DeploymentRequest(file_id="f1", metadata=_metadata(), horizon=MIN_FORECAST_HORIZON)
    DeploymentRequest(file_id="f1", metadata=_metadata(), horizon=MAX_FORECAST_HORIZON)
    for bad in (MIN_FORECAST_HORIZON - 1, MAX_FORECAST_HORIZON + 1):
        try:
            DeploymentRequest(file_id="f1", metadata=_metadata(), horizon=bad)
            raise AssertionError(f"horizon={bad} should have been rejected")
        except ValueError:
            pass


def test_deployment_request_default_matches_the_shared_constant():
    request = DeploymentRequest(file_id="f1", metadata=_metadata())
    assert request.horizon == DEFAULT_FORECAST_HORIZON


def test_estimation_request_uses_the_same_bounds_as_deployment():
    """The two requests describe the same run -- an estimate computed
    against a horizon the deploy step would reject is not a real estimate."""
    EstimationRequest(file_id="f1", metadata=_metadata(), horizon=MIN_FORECAST_HORIZON)
    EstimationRequest(file_id="f1", metadata=_metadata(), horizon=MAX_FORECAST_HORIZON)
    try:
        EstimationRequest(file_id="f1", metadata=_metadata(), horizon=MAX_FORECAST_HORIZON + 1)
        raise AssertionError("should have been rejected")
    except ValueError:
        pass
