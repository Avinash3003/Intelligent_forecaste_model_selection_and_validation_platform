"""Ray must be a declared engine dependency, not an inherited gift.

`execute_keys()` falls back to a sequential loop when Ray cannot be
imported, and does it silently — no exception, nothing in the run summary a
reader would notice, just the key-level parallelism gone and a run that
takes roughly four times as long on a 4-vCPU node.

For as long as the engine ran on a Databricks ML runtime that was
theoretical, because the runtime shipped Ray. A Databricks Container
Services image is built from `databricksruntime/standard`, which is not an
ML runtime and ships no Ray, so the declaration is now the only thing
standing between a container build and that silent fallback.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REQUIREMENTS = Path("forecast_engine/requirements.txt")
DOCKERFILE = Path("Dockerfile")

# The version the ML runtime this platform was measured on provides, so a
# container reproduces those runs rather than resolving to whatever is
# newest. Bump deliberately, together with the measurements.
EXPECTED_RAY_PIN = "ray==2.37.0"


def _requirement_lines() -> list[str]:
    return [
        line.split("#")[0].strip()
        for line in REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_ray_is_declared_in_the_engine_requirements():
    assert EXPECTED_RAY_PIN in _requirement_lines()


def test_ray_is_pinned_exactly_rather_than_floored():
    """A range would let a container resolve to a Ray the parallel path has
    never been measured against."""
    ray_lines = [line for line in _requirement_lines() if re.match(r"^ray\b", line)]

    assert ray_lines == [EXPECTED_RAY_PIN]


def test_ray_is_declared_exactly_once():
    """A second Ray requirement anywhere would let two pins disagree."""
    ray_lines = [line for line in _requirement_lines() if re.match(r"^ray\b", line)]

    assert len(ray_lines) == 1


@pytest.mark.skipif(not DOCKERFILE.exists(), reason="Dockerfile is not present")
def test_the_image_installs_ray_from_the_requirements_file_only():
    """The requirements file is the single source of the pin. A `pip install
    ray` in the Dockerfile would be a second place to keep in step, and the
    two would eventually drift."""
    dockerfile = DOCKERFILE.read_text()

    # Installed from files filtered out of the one requirements.txt (split
    # into two layers for push reliability — see the Dockerfile's own
    # comment), never from a second, separately-maintained list.
    assert "-r /tmp/requirements-core.txt" in dockerfile
    assert "-r /tmp/requirements-torch.txt" in dockerfile
    assert "grep" in dockerfile  # derives both files from requirements.txt
    # No direct install of ray as its own package.
    assert not re.search(r"pip install[^\n]*(?<![\w/-])ray[=<>\s]", dockerfile)


@pytest.mark.skipif(not DOCKERFILE.exists(), reason="Dockerfile is not present")
def test_the_image_installs_into_the_interpreter_databricks_actually_uses():
    """The base image's own `python3` is 3.10; Databricks runs tasks under
    the 3.11 virtualenv at /databricks/python3. Installing into the system
    interpreter produces an image whose packages are invisible at runtime."""
    assert "/databricks/python3" in DOCKERFILE.read_text()


@pytest.mark.skipif(not DOCKERFILE.exists(), reason="Dockerfile is not present")
def test_the_image_carries_no_application_source_or_secrets():
    """The image is the environment; the wheel is the application. Nothing
    copied in may be engine source, and no credential may reach a layer."""
    dockerfile = DOCKERFILE.read_text()

    copied = re.findall(r"^\s*COPY\s+(\S+)", dockerfile, flags=re.MULTILINE)
    assert copied == ["forecast_engine/requirements.txt"]

    lowered = dockerfile.lower()
    for secret in ("password", "client_secret", "azurecr.io", "token", "api_key"):
        assert secret not in lowered, f"Dockerfile must not mention {secret}"


def test_tft_is_pytorch_and_no_tensorflow_is_pulled_in():
    """TFT here is the Temporal Fusion Transformer from pytorch-forecasting.
    A TensorFlow package would be several hundred megabytes of an unrelated
    framework that nothing in the engine imports."""
    lines = _requirement_lines()
    joined = " ".join(lines).lower()

    assert any(line.startswith("pytorch-forecasting") for line in lines)
    assert any(line.startswith("torch") for line in lines)
    assert "tensorflow" not in joined
    assert "keras" not in joined
