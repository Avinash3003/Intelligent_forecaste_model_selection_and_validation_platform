"""Makes both packages importable without installing either.

`backend/` holds the `app` package and the repository root holds
`forecast_engine`; the tests exercise both, so both are put on the path
here rather than requiring an editable install of each.
"""

import os
import sys
from pathlib import Path

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
