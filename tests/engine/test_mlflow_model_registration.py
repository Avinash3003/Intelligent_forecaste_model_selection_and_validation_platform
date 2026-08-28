"""Model Registry compatibility across MLflow versions.

Production cloud execution resolves the pinned `mlflow>=2.15,<3.0`
to 2.x, whose `mlflow.pyfunc.log_model` has no `tags` parameter and no
`**kwargs` to absorb one — so passing it raised `TypeError` and every
registration failed. The local dev venv had MLflow 3.x, where `tags` is
accepted, which is exactly why it never surfaced outside the cloud.

These lock in the fix: registration never passes `tags` to `log_model`,
metadata is still applied (to the model version, whose API is identical on
both majors), and a tagging failure is not reported as a registration
failure.
"""

import inspect

import pytest

from forecast_engine.config.mlflow_config import MLflowConfig
from forecast_engine.core.pipeline_result import PipelineResult
from forecast_engine.s12_tracking.model_registrar import register_winner_models
from forecast_engine.utils.exceptions import MLflowTrackingError


class _ModelInfo:
    def __init__(self, version="3"):
        self.registered_model_version = version


class _FakeClient:
    """Records what registration actually sent to the SDK."""

    def __init__(self, tag_error=None, version="3"):
        self.log_calls = []
        self.tag_calls = []
        self._tag_error = tag_error
        self._version = version

    def register_pyfunc_model(self, python_model, artifact_path, registered_model_name, signature=None):
        self.log_calls.append(
            {
                "python_model": python_model,
                "artifact_path": artifact_path,
                "registered_model_name": registered_model_name,
                "signature": signature,
            }
        )
        return _ModelInfo(self._version)

    def set_model_version_tags(self, name, version, tags):
        if self._tag_error:
            raise MLflowTrackingError(self._tag_error)
        self.tag_calls.append({"name": name, "version": version, "tags": tags})


def _winner(group="1 | 1", model="prophet", fallback=False):
    return {
        "forecast_group": group,
        "final_production_model": model,
        "fallback_flag": fallback,
        "forecast": {"dates": ["2024-01-01"], "values": [10.0]},
    }


def _result(winners, dataset_path="store_item.csv"):
    return PipelineResult(
        run_id="run-1",
        dataset_metadata={"dataset_path": dataset_path} if dataset_path else {},
        final_winner_models=winners,
    )


def _register(client, winners, config=None):
    return register_winner_models(client, _result(winners), config or MLflowConfig(), "run-1")


# ---------------------------------------------------------------------
# The regression itself
# ---------------------------------------------------------------------


def test_the_installed_log_model_is_called_with_supported_arguments_only():
    """The actual bug: an argument the installed SDK does not accept.

    Checked against the real `mlflow.pyfunc.log_model` signature rather
    than a mock, so this fails on whichever MLflow version is installed if
    registration ever passes something it cannot take.
    """
    mlflow_pyfunc = pytest.importorskip("mlflow.pyfunc")
    accepted = set(inspect.signature(mlflow_pyfunc.log_model).parameters)

    client = _FakeClient()
    _register(client, [_winner()])

    sent = set(client.log_calls[0])
    unsupported = sent - accepted
    assert not unsupported, f"log_model does not accept {sorted(unsupported)} on this MLflow version"


def test_tags_are_never_passed_to_log_model():
    client = _FakeClient()
    _register(client, [_winner()])

    assert "tags" not in client.log_calls[0]


def test_the_facade_calls_the_real_sdk_with_arguments_it_accepts(monkeypatch):
    """The regression at its true boundary.

    The fake client above proves the registrar's contract; this proves the
    facade's. The recorded call is bound against the *installed*
    `log_model` signature, so on MLflow 2.x — where `tags` does not exist —
    reintroducing it fails here exactly as it failed in production, instead
    of passing locally on 3.x and breaking in the cloud.
    """
    mlflow_pyfunc = pytest.importorskip("mlflow.pyfunc")
    real_signature = inspect.signature(mlflow_pyfunc.log_model)
    captured = {}

    def _fake_log_model(*args, **kwargs):
        # Raises TypeError for an unsupported argument, just as the SDK does.
        real_signature.bind(*args, **kwargs)
        captured.update(kwargs)
        return _ModelInfo("1")

    monkeypatch.setattr(mlflow_pyfunc, "log_model", _fake_log_model)

    from forecast_engine.s12_tracking.mlflow_client import MLflowClient

    MLflowClient(MLflowConfig()).register_pyfunc_model(
        python_model=object(), artifact_path="model-1_1", registered_model_name="forecast_engine-1_1"
    )

    assert captured["registered_model_name"] == "forecast_engine-1_1"
    assert "tags" not in captured


def test_a_signature_is_sent_because_unity_catalog_requires_one():
    """Without this, registration reaches Unity Catalog and fails there:
    'Model passed for registration did not contain any signature metadata.'
    """
    client = _FakeClient()
    _register(client, [_winner()])

    assert client.log_calls[0]["signature"] is not None


def test_registration_succeeds_and_reports_a_version():
    results = _register(_FakeClient(version="7"), [_winner()])

    assert results[0].registered is True
    assert results[0].model_version == "7"
    assert results[0].error is None
    assert results[0].registered_at is not None


# ---------------------------------------------------------------------
# Metadata is preserved, not dropped
# ---------------------------------------------------------------------


def test_metadata_tags_are_still_applied():
    """The fix moves tags; it must not remove them."""
    client = _FakeClient()
    results = _register(client, [_winner(group="1 | 4", model="lightgbm")])

    assert len(client.tag_calls) == 1
    tags = client.tag_calls[0]["tags"]
    assert tags["forecast_group"] == "1 | 4"
    assert tags["model_name"] == "lightgbm"
    assert tags["fallback_used"] == "False"
    assert tags["run_id"] == "run-1"
    assert results[0].metadata_tags == tags


def test_tags_are_applied_to_the_version_that_was_just_registered():
    client = _FakeClient(version="12")
    _register(client, [_winner()])

    assert client.tag_calls[0]["version"] == "12"
    assert client.tag_calls[0]["name"] == client.log_calls[0]["registered_model_name"]


def test_a_fallback_winner_is_marked_as_one_in_the_registry():
    client = _FakeClient()
    _register(client, [_winner(model="seasonal_naive", fallback=True)])

    assert client.tag_calls[0]["tags"]["fallback_used"] == "True"


# ---------------------------------------------------------------------
# The registered model corresponds to the selected winner
# ---------------------------------------------------------------------


def test_the_registered_model_is_the_final_selected_winner():
    client = _FakeClient()
    winners = [_winner("1 | 1", "prophet"), _winner("1 | 2", "seasonal_naive", fallback=True)]

    results = _register(client, winners)

    registered = {r.group_id: r.model_name for r in results}
    assert registered == {"1 | 1": "prophet", "1 | 2": "seasonal_naive"}
    # And the wrapper carries that same model, not another group's.
    for call in client.log_calls:
        wrapper = call["python_model"]
        assert registered[wrapper.group_id] == wrapper.model_name


def test_every_key_in_one_dataset_shares_one_registered_model():
    """A per-key name would mean a 500-key dataset creates 500 permanent
    registry entries on its first run alone — every key registers as a
    new version of one model instead."""
    client = _FakeClient()
    _register(client, [_winner("1 | 1"), _winner("1 | 2"), _winner("1 | 3")])

    names = {call["registered_model_name"] for call in client.log_calls}
    assert len(names) == 1


def test_different_datasets_get_different_registered_models():
    """Two unrelated datasets must never share a registered model just
    because they happen to have a same-named key."""
    client = _FakeClient()
    register_winner_models(
        client, _result([_winner("1 | 1")], dataset_path="store_a.csv"), MLflowConfig(), "run-1"
    )
    register_winner_models(
        client, _result([_winner("1 | 1")], dataset_path="store_b.csv"), MLflowConfig(), "run-2"
    )

    names = {call["registered_model_name"] for call in client.log_calls}
    assert len(names) == 2


def test_a_dataset_with_no_recorded_path_falls_back_to_the_run_id():
    client = _FakeClient()
    register_winner_models(client, _result([_winner("1 | 1")], dataset_path=None), MLflowConfig(), "run-xyz")

    assert "run-xyz" in client.log_calls[0]["registered_model_name"]


def test_a_group_with_no_selected_model_is_not_registered():
    client = _FakeClient()
    winner = _winner()
    winner["final_production_model"] = None

    results = _register(client, [winner])

    assert client.log_calls == []
    assert results[0].registered is False


# ---------------------------------------------------------------------
# Failure reporting stays honest
# ---------------------------------------------------------------------


def test_a_tagging_failure_does_not_make_a_registration_look_failed():
    """The model is in the registry; only its annotation is missing."""
    client = _FakeClient(tag_error="registry rejected the tag")
    results = _register(client, [_winner()])

    assert results[0].registered is True
    assert results[0].error is None
    assert "registry rejected the tag" in results[0].tag_error
    assert results[0].metadata_tags == {}


def test_one_group_failing_to_register_does_not_stop_the_next():
    class _FlakyClient(_FakeClient):
        def register_pyfunc_model(self, python_model, artifact_path, registered_model_name, signature=None):
            if python_model.group_id == "1 | 1":
                raise MLflowTrackingError("boom")
            return super().register_pyfunc_model(python_model, artifact_path, registered_model_name, signature)

    results = _register(_FlakyClient(), [_winner("1 | 1"), _winner("1 | 2")])

    assert results[0].registered is False and "boom" in results[0].error
    assert results[1].registered is True


def test_registration_can_be_disabled():
    client = _FakeClient()
    results = _register(client, [_winner()], MLflowConfig(register_winner_model=False))

    assert results == []
    assert client.log_calls == []
