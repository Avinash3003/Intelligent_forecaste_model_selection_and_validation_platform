"""Artifacts mirror: a blob-accessible copy of business insights and the
LLM trace, alongside MLflow's own copy of the same data.
"""

import json

from forecast_engine.config.pipeline_config import ArtifactsMirrorConfig
from forecast_engine.s03_storage.artifacts_mirror_writer import ArtifactsMirrorWriter


def _writer(tmp_path):
    return ArtifactsMirrorWriter(ArtifactsMirrorConfig(root_dir=str(tmp_path / "artifacts")))


def test_business_insights_are_mirrored_as_json(tmp_path):
    insights = {"available": True, "status": "generated", "groups": {"1 | 1": {}}}
    result = _writer(tmp_path).write(insights, {}, "run-1")

    entry = next(p for p in result["persisted"] if p["file"] == "business_insights.json")
    assert entry["persisted"] is True
    assert json.loads(open(entry["uri"]).read()) == insights


def test_llm_trace_is_mirrored_only_when_it_has_calls(tmp_path):
    trace = {"run_id": "run-1", "summary": {"call_count": 1}, "calls": [{"group_id": "1 | 1"}]}
    result = _writer(tmp_path).write({}, trace, "run-1")

    entry = next(p for p in result["persisted"] if p["file"] == "llm_trace.json")
    assert entry["persisted"] is True
    assert json.loads(open(entry["uri"]).read()) == trace


def test_an_empty_trace_is_not_mirrored(tmp_path):
    result = _writer(tmp_path).write({"available": False}, {}, "run-1")

    files = [p["file"] for p in result["persisted"]]
    assert "llm_trace.json" not in files


def test_no_insights_and_no_trace_mirrors_nothing(tmp_path):
    result = _writer(tmp_path).write({}, {}, "run-1")

    assert result["persisted"] == []


def test_both_files_land_under_the_same_run_directory(tmp_path):
    insights = {"available": True}
    trace = {"calls": [{"group_id": "1 | 1"}]}
    result = _writer(tmp_path).write(insights, trace, "run-1")

    uris = {p["uri"] for p in result["persisted"]}
    assert all(str(tmp_path / "artifacts" / "run-1") in uri for uri in uris)


def test_different_runs_do_not_collide(tmp_path):
    insights = {"available": True}
    a = _writer(tmp_path).write(insights, {}, "run-a")
    b = _writer(tmp_path).write(insights, {}, "run-b")

    assert a["persisted"][0]["uri"] != b["persisted"][0]["uri"]


def test_a_malicious_run_id_cannot_escape_the_artifacts_directory(tmp_path):
    result = _writer(tmp_path).write({"available": True}, {}, "../../escape")

    uri = result["persisted"][0]["uri"]
    assert uri.startswith(str(tmp_path / "artifacts"))
    assert ".." not in uri


def test_disabled_mirror_writes_nothing(tmp_path):
    writer = ArtifactsMirrorWriter(ArtifactsMirrorConfig(enabled=False, root_dir=str(tmp_path / "artifacts")))
    result = writer.write({"available": True}, {"calls": [{}]}, "run-1")

    assert result == {"enabled": False, "persisted": []}
    assert not (tmp_path / "artifacts").exists()
