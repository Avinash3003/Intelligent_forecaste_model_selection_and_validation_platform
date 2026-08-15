"""Persists the model that won for each forecast key.

So a run's decision can be reloaded and served later without re-running the
pipeline. Three deliberate properties:

  - Only winners are written; persisting every candidate would multiply
    storage to keep models no decision refers to.
  - Nothing is retrained — the fitted wrapper is already on the training
    record, and this only serializes it.
  - Storage is a location, not a mechanism: the caller supplies an absolute
    root and this joins paths beneath it, so no cloud SDK or credential
    reaches the engine.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from forecast_engine.config.pipeline_config import ModelStorageConfig

# Anything outside this set becomes "_". Applied to the *whole* key, so a
# key containing a path separator or "..'" cannot escape the run directory —
# the separators simply stop being separators.
_UNSAFE_KEY_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")

# Two or more consecutive dots, which must never survive into a filename.
_DOT_RUNS = re.compile(r"\.{2,}")


def sanitize_forecast_key(raw: str) -> str:
    """A forecast key made safe for a filename.

    Traversal is prevented structurally, not by blocklist: every separator
    is replaced, so the result is always one path segment.
    """
    cleaned = _UNSAFE_KEY_CHARS.sub("_", str(raw))
    # Collapse runs of dots. Replacing separators alone already makes the
    # result a single segment, so this is not what prevents traversal — it
    # keeps ".." from surviving *inside* a name, where it reads as an escape
    # attempt to anything later inspecting these paths.
    cleaned = _DOT_RUNS.sub("_", cleaned).strip("_.")
    if not cleaned:
        return "key"
    return cleaned


class WinningModelWriter:
    """Serializes the winning fitted model for each forecast key."""

    def __init__(self, config: ModelStorageConfig | None = None) -> None:
        self._config = config or ModelStorageConfig()

    def write_all(
        self,
        winners: list[Any],
        trained_models: list[Any],
        run_id: str,
    ) -> list[dict[str, Any]]:
        """Write one model per winning key.

        Returns a record per winner, including failures, so the run summary
        reports persistence honestly instead of omitting a key.
        """
        if not self._config.enabled:
            return []

        # (group, model) -> fitted wrapper. Built once: a run can have
        # thousands of trained records, and scanning them per winner would
        # make this quadratic in key count.
        fitted = {
            (record.group_id, record.model_name): record.fitted_model
            for record in trained_models
            if getattr(record, "fitted_model", None) is not None
        }

        run_dir = Path(self._config.root_dir) / sanitize_forecast_key(run_id)
        results: list[dict[str, Any]] = []
        for winner in winners:
            results.append(self._write_one(winner, fitted, run_dir))
        return results

    def _write_one(
        self, winner: Any, fitted: dict[tuple[str, str], Any], run_dir: Path
    ) -> dict[str, Any]:
        group_id = getattr(winner, "group_id", None)
        model_name = getattr(winner, "model_name", None)

        record: dict[str, Any] = {
            "forecast_group": group_id,
            "model_name": model_name,
            "persisted": False,
            "uri": None,
            "error": None,
        }

        if not group_id or not model_name:
            # A group with no production model is a legitimate outcome, not
            # a failure — there is simply nothing to persist.
            status = getattr(getattr(winner, "status", None), "value", None)
            record["error"] = (
                f"No production model was selected for this group (Final Selection Status: {status})."
            )
            return record

        # A ranked winner's estimator lives on its training record; a
        # fallback winner's was fitted during selection and is carried on the
        # result itself. Either way it is already fitted — nothing is
        # retrained to serialize it.
        model = fitted.get((group_id, model_name)) or getattr(winner, "fitted_model", None)
        if model is None:
            record["error"] = "The winning model's fitted estimator was not available in this run."
            return record

        path = run_dir / f"{sanitize_forecast_key(group_id)}_model.pkl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:  # noqa: BLE001 - one key must not end the run
            record["error"] = f"Could not write the model: {exc}"
            return record

        record["persisted"] = True
        record["uri"] = str(path)
        return record
