"""LLM Evaluation & Regression report API (Section 13.3) — read-only
exposure of whatever `python -m forecast_engine.s11_llm.evaluate` last
wrote. This route never runs an evaluation itself.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_principal
from app.auth.models import Principal, Role
from app.auth.rbac import permissions_for
from app.config.settings import Settings, get_settings
from app.main import app
from app.services.llm_evaluation_service import LlmEvaluationService, get_llm_evaluation_service


def _principal(role: Role) -> Principal:
    return Principal(subject=f"user-{role.value}", display_name=role.value, roles=[role], permissions=permissions_for([role]))


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


_SAMPLE_REPORT = {
    "dataset_version": "v1",
    "prompt_version": "v2",
    "generation_mode": "template",
    "case_count": 2,
    "generated_count": 2,
    "schema_pass_rate": 1.0,
    "groundedness_rate": 1.0,
    "winner_consistency_rate": 1.0,
    "rejection_accuracy_rate": 1.0,
    "hallucination_rate": 0.0,
    "readability_pass_rate": 1.0,
    "overall_pass_rate": 1.0,
    "thresholds": {
        "minimum_schema_pass_rate": 0.9,
        "minimum_groundedness": 0.85,
        "minimum_winner_consistency": 1.0,
        "minimum_rejection_accuracy": 0.8,
        "maximum_hallucination_rate": 0.15,
        "minimum_readability_pass_rate": 0.85,
    },
    "regression_passed": True,
    "threshold_violations": [],
    "results": [
        {
            "case_id": "clean_win",
            "scenario": "normal_winner",
            "expected": {"selected_model": "xgboost", "wmape": 8.2, "is_fallback": False, "rejected_candidates": []},
            "insight": {
                "selected_model": "xgboost", "rejection_reasons": [], "confidence": 91.8,
                "caveats": [], "concise_summary": "xgboost was selected, with a backtest WMAPE of 8.2%.",
            },
            "hallucination_category": "grounded",
            "checks": {
                "schema_validity": {"passed": True, "detail": ""},
                "groundedness": {"passed": True, "detail": ""},
            },
            "failed_checks": [],
            "generation_error": None,
            "overall": "PASS",
        }
    ],
}


def test_unauthenticated_caller_cannot_read_the_report(client):
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update={"auth_enabled": True})
    assert client.get("/results/llmops/evaluation").status_code == 401


def test_analyst_can_read_the_evaluation_report(client):
    # Read-only exposure of a report file, gated the same way as the rest
    # of the Observability page — the Analyst role can view it.
    app.dependency_overrides[get_current_principal] = lambda: _principal(Role.ANALYST)
    assert client.get("/results/llmops/evaluation").status_code == 200


def test_no_report_file_yet_is_reported_honestly_not_as_an_error(client, tmp_path):
    app.dependency_overrides[get_current_principal] = lambda: _principal(Role.DATA_SCIENTIST)
    settings = Settings(llm_eval_report_path=str(tmp_path / "does_not_exist.json"))
    app.dependency_overrides[get_llm_evaluation_service] = lambda: LlmEvaluationService(settings)

    response = client.get("/results/llmops/evaluation")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["unavailable_reason"]
    assert body["results"] == []
    assert body["case_count"] == 0


def test_an_existing_report_is_read_back_verbatim(client, tmp_path):
    app.dependency_overrides[get_current_principal] = lambda: _principal(Role.DATA_SCIENTIST)
    report_path = tmp_path / "latest_regression_report.json"
    report_path.write_text(json.dumps(_SAMPLE_REPORT))
    settings = Settings(llm_eval_report_path=str(report_path))
    app.dependency_overrides[get_llm_evaluation_service] = lambda: LlmEvaluationService(settings)

    response = client.get("/results/llmops/evaluation")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["dataset_version"] == "v1"
    assert body["prompt_version"] == "v2"
    assert body["case_count"] == 2
    assert body["overall_pass_rate"] == 1.0
    assert body["regression_passed"] is True
    assert body["generated_at"] is not None
    assert len(body["results"]) == 1
    assert body["results"][0]["case_id"] == "clean_win"
    assert body["results"][0]["expected"]["selected_model"] == "xgboost"


def test_a_corrupted_report_file_is_reported_as_unavailable_not_a_500(client, tmp_path):
    app.dependency_overrides[get_current_principal] = lambda: _principal(Role.DATA_SCIENTIST)
    report_path = tmp_path / "latest_regression_report.json"
    report_path.write_text("{not valid json")
    settings = Settings(llm_eval_report_path=str(report_path))
    app.dependency_overrides[get_llm_evaluation_service] = lambda: LlmEvaluationService(settings)

    response = client.get("/results/llmops/evaluation")

    assert response.status_code == 200
    assert response.json()["available"] is False


def test_settings_resolve_a_default_report_path_when_unconfigured(tmp_path):
    settings = Settings(forecast_engine_root=str(tmp_path / "forecast_engine"))
    resolved = settings.llm_eval_report_path_resolved
    assert resolved.name == "latest_regression_report.json"
    assert "s11_llm" in resolved.parts
    assert "eval_output" in resolved.parts


def test_settings_prefer_an_explicit_override_path(tmp_path):
    custom = tmp_path / "custom_report.json"
    settings = Settings(llm_eval_report_path=str(custom))
    assert settings.llm_eval_report_path_resolved == custom
