# forecast_engine — dependencies-only runtime image for Databricks Container Services.
# App code is deployed separately via Databricks Asset Bundles, not copied here.

# Databricks-supported base for DBR 15.4 LTS — has the JDK/bash/sudo DCS requires.
FROM databricksruntime/python:15.4-LTS

# Skip .pyc files, stream logs unbuffered.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Deps must install into this interpreter — it's the one DCS runs jobs with.
ENV PIP=/databricks/python3/bin/pip

# Copy only the dependency file first, so the cache holds until it changes.
COPY forecast_engine/requirements.txt /tmp/requirements.txt

# Update pip itself.
RUN $PIP install --no-cache-dir --upgrade pip

# Install every dependency (all prebuilt wheels — xgboost-cpu, CPU-only torch).
RUN $PIP install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Confirm every dependency imports in the same interpreter DCS uses.
# Timeout is 30s: this import chain takes ~22s cold, measured not guessed.
HEALTHCHECK --interval=60s --timeout=30s --start-period=40s --retries=3 \
    CMD /databricks/python3/bin/python -c "import mlflow, numpy, openai, pandas, prophet, pytorch_forecasting, scipy, shap, sklearn, statsmodels, torch, xgboost, lightgbm" || exit 1

# No USER/CMD/ENTRYPOINT override — DCS starts the container as root with its
# own entrypoint, so setting either here would break cluster startup.
