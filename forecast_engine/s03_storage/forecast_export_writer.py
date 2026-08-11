"""Forecast export — the business-facing forecast values, written durably.

Distinct from the other two storage writers in this package: the curated
writer persists the *input* the models trained on, and the model writer
persists the *model* itself; this persists the *output* — what every group
was actually forecast to be, over the horizon, with its confidence bounds.
A business user exporting a run's numbers wants this file, not a `.pkl` or
a training dataset.

One CSV per run, every group's forecast as its own rows — not one file per
group, so a run's complete export is always a single download regardless of
key count.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from forecast_engine.config.pipeline_config import ForecastExportConfig
from forecast_engine.s03_storage.model_writer import sanitize_forecast_key

_COLUMNS = ("group_id", "model_name", "date", "value", "lower", "upper")


class ForecastExportWriter:
    """Writes one run's complete forecast output as a single CSV."""

    def __init__(self, config: ForecastExportConfig | None = None) -> None:
        self._config = config or ForecastExportConfig()

    def write(self, winners: list[Any], run_id: str) -> dict[str, Any]:
        """Persist every winning group's forecast as one CSV.

        Returns a record describing what was written — never raises: a
        run whose forecasts could not be exported is still a complete,
        correct run, so export failure is reported, not fatal.
        """
        if not self._config.enabled:
            return {"enabled": False, "persisted": False, "uri": None, "rows": 0, "error": None}

        rows = list(self._rows(winners))
        if not rows:
            return {
                "enabled": True,
                "persisted": False,
                "uri": None,
                "rows": 0,
                "error": "No group produced a forecast to export.",
            }

        path = Path(self._config.root_dir) / f"{sanitize_forecast_key(run_id)}_forecast.csv"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(_COLUMNS)
            writer.writerows(rows)
            path.write_text(buffer.getvalue(), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - one write failure must not fail the run
            return {"enabled": True, "persisted": False, "uri": None, "rows": 0, "error": f"Could not write the export: {exc}"}

        return {"enabled": True, "persisted": True, "uri": str(path), "rows": len(rows), "error": None}

    def _rows(self, winners: list[Any]):
        for winner in winners:
            forecast = getattr(winner, "forecast", None)
            if forecast is None:
                continue
            dates = forecast.dates or []
            values = forecast.values or []
            lower = forecast.lower or [None] * len(values)
            upper = forecast.upper or [None] * len(values)
            for date, value, low, high in zip(dates, values, lower, upper):
                yield (winner.group_id, winner.model_name, date, value, low, high)
