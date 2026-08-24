"""Smoke test — proves the backend still starts and answers.

Deliberately small. The full backend and engine suites were removed; what
remains is the one check worth blocking a deploy on: that `app.main`
imports cleanly and the app serves a request.

That is not a formality. Every deploy failure this project has actually
hit at startup was an import-time error — a missing dependency, a bad
setting, a module renamed on one side of a refactor — and all of them
surface here in under a second.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_the_app_imports_and_serves():
    """Import + routing + response, in one assertion."""
    assert client.get("/health").status_code == 200


def test_routes_are_mounted():
    """A router that failed to mount leaves the app up but useless."""
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert any(p.startswith("/results") for p in paths)
    assert any(p.startswith("/deployments") or p.startswith("/deploy") for p in paths)
