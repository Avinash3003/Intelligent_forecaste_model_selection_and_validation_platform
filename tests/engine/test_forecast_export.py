"""Forecast export: the run's business-facing output, written as one CSV.

Distinct from the curated writer (persists the input) and the model writer
(persists the model) — this persists what every group was actually
forecast to be, so a user can download the numbers without opening MLflow
or a `.pkl`.
"""

import csv
from dataclasses import dataclass, field

from forecast_engine.config.pipeline_config import ForecastExportConfig
from forecast_engine.s03_storage.forecast_export_writer import ForecastExportWriter


@dataclass
class _Forecast:
    dates: list = field(default_factory=list)
    values: list = field(default_factory=list)
    lower: list | None = None
    upper: list | None = None


@dataclass
class _Winner:
    group_id: str
    model_name: str | None = "prophet"
    forecast: _Forecast | None = None


def _writer(tmp_path):
    return ForecastExportWriter(ForecastExportConfig(root_dir=str(tmp_path / "forecasts")))


def _read_rows(uri):
    with open(uri, newline="") as handle:
        return list(csv.reader(handle))


def test_every_group_and_every_date_becomes_a_row(tmp_path):
    winners = [
        _Winner("1 | 1", forecast=_Forecast(["2024-01", "2024-02"], [10.0, 12.0], [8.0, 9.0], [12.0, 15.0])),
        _Winner("1 | 2", forecast=_Forecast(["2024-01"], [5.0], [4.0], [6.0])),
    ]
    result = _writer(tmp_path).write(winners, "run-1")

    assert result["persisted"] is True
    assert result["rows"] == 3
    rows = _read_rows(result["uri"])
    assert rows[0] == ["group_id", "model_name", "date", "value", "lower", "upper"]
    assert rows[1] == ["1 | 1", "prophet", "2024-01", "10.0", "8.0", "12.0"]
    assert rows[3] == ["1 | 2", "prophet", "2024-01", "5.0", "4.0", "6.0"]


def test_a_group_with_no_forecast_is_skipped_not_a_blank_row(tmp_path):
    winners = [_Winner("1 | 1", forecast=None), _Winner("1 | 2", forecast=_Forecast(["2024-01"], [5.0]))]
    result = _writer(tmp_path).write(winners, "run-1")

    assert result["rows"] == 1


def test_missing_bounds_are_written_as_empty_not_fabricated(tmp_path):
    winners = [_Winner("1 | 1", forecast=_Forecast(["2024-01"], [10.0], None, None))]
    result = _writer(tmp_path).write(winners, "run-1")

    rows = _read_rows(result["uri"])
    assert rows[1] == ["1 | 1", "prophet", "2024-01", "10.0", "", ""]


def test_no_winners_with_a_forecast_reports_nothing_exported(tmp_path):
    result = _writer(tmp_path).write([_Winner("1 | 1", forecast=None)], "run-1")

    assert result["persisted"] is False
    assert result["rows"] == 0
    assert "No group produced a forecast" in result["error"]


def test_one_file_covers_the_whole_run_not_one_per_group(tmp_path):
    winners = [_Winner(f"1 | {i}", forecast=_Forecast(["2024-01"], [1.0])) for i in range(5)]
    result = _writer(tmp_path).write(winners, "run-1")

    csv_files = list((tmp_path / "forecasts").rglob("*.csv"))
    assert len(csv_files) == 1
    assert result["rows"] == 5


def test_different_runs_do_not_overwrite_each_other(tmp_path):
    winner = _Winner("1 | 1", forecast=_Forecast(["2024-01"], [1.0]))
    a = _writer(tmp_path).write([winner], "run-a")
    b = _writer(tmp_path).write([winner], "run-b")

    assert a["uri"] != b["uri"]


def test_a_malicious_run_id_cannot_escape_the_export_directory(tmp_path):
    winner = _Winner("1 | 1", forecast=_Forecast(["2024-01"], [1.0]))
    result = _writer(tmp_path).write([winner], "../../escape")

    assert result["uri"].startswith(str(tmp_path / "forecasts"))
    assert ".." not in result["uri"]


def test_disabled_export_writes_nothing(tmp_path):
    writer = ForecastExportWriter(ForecastExportConfig(enabled=False, root_dir=str(tmp_path / "forecasts")))
    result = writer.write([_Winner("1 | 1", forecast=_Forecast(["2024-01"], [1.0]))], "run-1")

    assert result == {"enabled": False, "persisted": False, "uri": None, "rows": 0, "error": None}
    assert not (tmp_path / "forecasts").exists()
