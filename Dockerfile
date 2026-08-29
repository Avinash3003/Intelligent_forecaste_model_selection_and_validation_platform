# ForecastIQ runtime environment for Databricks Container Services.
#
# This image is the ENVIRONMENT, never the application. The application is
# the forecast_engine wheel, which every run installs through the job's
# `libraries` list — keeping the two apart means shipping new engine code is
# a wheel rebuild, not a multi-gigabyte image rebuild and re-push.
#
# Base image: databricksruntime/standard, which is deliberately NOT an ML
# runtime. Databricks publishes no ML base image for Container Services
# (the published set is standard / python / minimal / rbase / gpu-*), so
# everything the ML runtime used to hand us for free — Ray above all — is
# installed here from forecast_engine/requirements.txt instead.
FROM databricksruntime/standard:15.4-LTS

# Databricks does not run tasks under the image's system interpreter. This
# image ships python3 = 3.10, while /databricks/python3 is a 3.11 virtualenv
# and is what a job actually executes under, so anything installed anywhere
# else is invisible at runtime. ARG rather than ENV: this is a build-time
# detail and has no business leaking into the running container's env.
ARG DBX_PYTHON=/databricks/python3/bin/python

# Only the requirements file enters the image. Engine source is excluded on
# purpose (see .dockerignore) so a code change cannot invalidate this layer.
COPY forecast_engine/requirements.txt /tmp/requirements.txt

# Split into two installs, both still resolved from the one requirements
# file — this changes nothing about which versions get installed, only
# which Docker layer they land in.
#
# torch + pytorch-forecasting are the large half of this image (the CPU
# wheel plus its own dependency tree), and a registry push transfers one
# Docker layer as one blob. On a link that cannot sustain a very long
# transfer, a single ~2GB layer is the one most likely to be interrupted
# partway through and have to restart from zero. Splitting it into its own
# layer caps the size any one blob transfer has to survive.
RUN grep -E '^--extra-index-url|^(torch|pytorch-forecasting)([<>=~]|$)' /tmp/requirements.txt > /tmp/requirements-torch.txt \
 && grep -Ev '^(torch|pytorch-forecasting)([<>=~]|$)' /tmp/requirements.txt > /tmp/requirements-core.txt

# The file carries its own --extra-index-url for the CPU-only torch build,
# so the resolve here is identical to a laptop's and no CUDA packages are
# pulled — tft_model.py hardcodes accelerator="cpu".
RUN "$DBX_PYTHON" -m pip install --no-cache-dir -r /tmp/requirements-core.txt

RUN "$DBX_PYTHON" -m pip install --no-cache-dir -r /tmp/requirements-torch.txt \
 && rm -f /tmp/requirements*.txt

# Build-time gate. execute_keys() falls back to a sequential loop when Ray
# cannot be imported, and that fallback is silent — no error, no warning in
# the product, just the key-level parallelism quietly gone. An image that
# would do that must fail here rather than in production.
RUN "$DBX_PYTHON" - <<'PY'
import importlib

REQUIRED = (
    "ray",                    # key-parallel execution — the silent-fallback risk
    "pandas", "numpy", "scipy", "sklearn", "openpyxl",
    "statsmodels", "xgboost", "lightgbm", "prophet",
    "torch", "pytorch_forecasting",   # TFT (PyTorch, not TensorFlow)
    "shap", "mlflow", "openai",
    # The Files API client. This image has no UC Volumes mount, so it is
    # how a container run copies its outputs into the storage account —
    # see forecast_engine/s03_storage/volume_sync.py. It arrives
    # transitively via mlflow today, which is exactly why it is asserted
    # here: a future mlflow that stopped pulling it would otherwise fail
    # the run, not the build.
    "databricks.sdk",
)

for name in REQUIRED:
    importlib.import_module(name)

import ray
assert ray.__version__ == "2.37.0", f"expected ray 2.37.0, got {ray.__version__}"
print("runtime environment OK — ray", ray.__version__)
PY
