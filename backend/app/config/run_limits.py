"""Business-rule bounds shared by every backend surface that accepts a run
configuration.

The single source of truth for these numbers is the engine itself
(`forecast_engine/run_pipeline.py`'s `MIN_FORECAST_HORIZON` /
`MAX_FORECAST_HORIZON`) — it is what actually enforces them at execution
time, and a request the backend accepted but the engine then rejected would
be a validation gap, not merely an inconsistency.

The engine cannot be imported here to read them directly: it is a separate
package with its own venv (torch, ray and the rest are not on this
process's path), the same reason `mlflow_history.py` and
`estimation_service.py` already duplicate constants from it as plain
values rather than imports. Duplicated the same way, and the duplication is
what `tests/backend/test_run_limits_match_the_engine.py` checks by reading
the engine's own source text — a value changed on one side without the
other will fail that test, not drift silently.
"""

from __future__ import annotations

MIN_FORECAST_HORIZON = 6
MAX_FORECAST_HORIZON = 60

# The platform's own default, distinct from the bounds above: Section 3's
# stated minimum recommended horizon, not the floor the engine will accept.
DEFAULT_FORECAST_HORIZON = 12
