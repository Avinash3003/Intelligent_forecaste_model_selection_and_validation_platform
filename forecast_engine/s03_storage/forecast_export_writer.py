"""The business-facing forecast values.

The other writers persist the input (curated) and the model; this persists
the output — what each group was forecast to be, with bounds. This is the
file a business user actually wants.

One CSV per run with every group's rows in it, so a complete export is a
single download regardless of key count.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from forecast_engine.config.pipeline_config import ForecastExportConfig
from forecast_engine.core import storage
from forecast_engine.s03_storage.model_writer import sanitize_forecast_key

_COLUMNS = ("group_id", "model_name", "date", "value", "lower", "upper")


class ForecastExportWriter:
    """Writes one run's complete forecast output as a single CSV."""

    def __init__(self, config: ForecastExportConfig | None = None) -> None:
        self._config = config or ForecastExportConfig()

    def write(self, winners: list[Any], run_id: str) -> dict[str, Any]:
        """Write every winner's forecast to one CSV.

        Never raises: a run whose export failed is still a correct run, so
        the failure is reported on the returned record.
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
            storage.ensure_parent(path)
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(_COLUMNS)
            writer.writerows(rows)
            storage.write_text(path, buffer.getvalue())
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
