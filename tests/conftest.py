"""Makes both packages importable without installing either.

`backend/` holds the `app` package and the repository root holds
`forecast_engine`; the tests exercise both, so both are put on the path
here rather than requiring an editable install of each.
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# `Settings` resolves FORECAST_ENGINE_ROOT relative to the process's working
# directory, and its default assumes the backend is started from `backend/`.
# Pytest runs from the repository root, so it is pinned here — otherwise
# merely importing `app.main` fails on a path that does not exist.
os.environ.setdefault("FORECAST_ENGINE_ROOT", str(ROOT / "forecast_engine"))
# Importing `app.main` constructs the Pipeline Executor, and LocalRunner
# refuses to build when the engine's own interpreter is missing. That
# interpreter is a developer's local venv, which no CI runner has — so
# collection failed there while passing on a developer machine. No test
# launches the engine subprocess, so pointing this at the interpreter
# already running the suite satisfies the check without changing what is
# exercised.
os.environ.setdefault("FORECAST_ENGINE_PYTHON", sys.executable)
# Never write staged uploads into the working tree during a test run.
os.environ.setdefault("UPLOAD_DIR", str(ROOT / ".pytest-uploads"))

# The suite must never read a developer's real `.env`.
#
# `Settings.model_config` sets env_file=".env", which pydantic-settings
# resolves against the *working directory*. From the repository root that
# names nothing, so the suite ran on its own fixtures; from `backend/` it
# names the developer's live configuration, and 14 tests failed on values
# no test set. Worse than the noise is the direction it can fail in: real
# credentials silently satisfying a test that is supposed to prove the code
# supplies them, which is how a missing MLflow URI and missing Azure OpenAI
# credentials both passed review this way.
#
# Pinned to None (pydantic-settings for "no env file") rather than to a
# path, so the answer cannot depend on where pytest was started from.
#
# Guarded because this same conftest serves tests/engine, which runs under
# the engine's own interpreter — that venv has no pydantic-settings, and no
# engine test constructs Settings.
try:  # noqa: SIM105 - the except needs a comment, contextlib.suppress cannot carry one
    from app.config.settings import Settings  # noqa: E402 - needs sys.path above

    Settings.model_config["env_file"] = None
except ImportError:
    pass


@pytest.fixture(scope="session")
def _migrated_mlflow_db(tmp_path_factory) -> Path:
    """One migrated MLflow SQLite store, built once for the whole session.

    MLflow runs its Alembic migrations the first time it connects to a new
    SQLite file — measured at ~4s, and paid again for every new file, since
    nothing about it is cached across databases. A suite that gives each
    test a fresh store therefore spends most of its runtime migrating
    schemas rather than exercising code.

    Built once here and copied per test by `mlflow_db` below. Importing
    mlflow lazily keeps a suite that never touches it free of the cost.
    """
    path = tmp_path_factory.mktemp("mlflow-template") / "mlflow.db"
    from mlflow.tracking import MlflowClient

    # Any store read initializes the schema; this is the cheapest one.
    MlflowClient(tracking_uri=f"sqlite:///{path}").search_experiments()
    return path


@pytest.fixture
def mlflow_db(_migrated_mlflow_db, tmp_path) -> Path:
    """A per-test MLflow SQLite store that is already migrated.

    Isolation is identical to creating one from scratch — every test still
    gets its own file, under its own `tmp_path`, and may write to it
    freely — but the migrations are copied rather than re-run.
    """
    path = tmp_path / "mlflow.db"
    shutil.copyfile(_migrated_mlflow_db, path)
    return path
