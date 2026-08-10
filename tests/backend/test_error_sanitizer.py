"""Nothing that reaches a client may carry a secret, a token or an endpoint.

These tests are written against the failure text real infrastructure
produces, not invented strings.
"""

from app.utils.errors import GENERIC_MESSAGE, friendly_message, redact, safe_detail


def test_databricks_secret_failure_becomes_an_actionable_message():
    raw = (
        "INVALID_PARAMETER_VALUE: X_SecretResolutionFailure: Failed to resolve secret "
        "reference {{secrets/forecastiq/azure-openai-api-key}}"
    )
    message = safe_detail(raw)

    assert "credential" in message.lower()
    assert "X_SecretResolutionFailure" not in message
    assert "secrets/" not in message


def test_workspace_urls_never_survive_redaction():
    raw = "Failed calling https://adb-1234567890123456.4.azuredatabricks.net/api/2.2/jobs/run-now"
    assert "adb-1234567890123456" not in redact(raw)
    assert "azuredatabricks.net" not in redact(raw)


def test_tokens_and_keys_are_redacted():
    fake_token = "dapi" + "0" * 32  # shape-only stand-in, not a real token
    assert "dapi" not in redact(f"auth failed for {fake_token}")
    assert "eyJhbGciOi" not in redact("Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.abcdefghij")
    assert "supersecret" not in redact("AccountKey=supersecretvalue123==;EndpointSuffix=core.windows.net")


def test_server_paths_are_redacted():
    assert "/home/sigmoid" not in redact("FileNotFoundError: /home/sigmoid/Documents/tech_demo/uploads/x.csv")


def test_unrecognised_errors_fall_back_rather_than_echo():
    # The critical property: an error we cannot classify is exactly the one
    # we must not repeat back verbatim.
    raw = "RuntimeError: internal state 0xdeadbeef in /opt/app/secrets.py"
    assert friendly_message(raw) == GENERIC_MESSAGE
    assert "0xdeadbeef" not in safe_detail(raw)


def test_quota_failure_is_explained_as_capacity():
    raw = "CLOUD_PROVIDER_RESOURCE_STOCKOUT: SkuNotAvailable for Standard_D4s_v5"
    assert "capacity" in safe_detail(raw).lower()


def test_safe_detail_accepts_exceptions_as_well_as_strings():
    assert safe_detail(TimeoutError("connection timed out")) == safe_detail("connection timed out")
