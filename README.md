# Intelligent Forecast Model Selection & Validation Platform

Enterprise Auto-ML time-series forecasting. A business user uploads a dataset, confirms its
column roles, picks candidate models, and triggers a run. The platform trains every candidate per
business key, backtests them, screens their forward forecasts for statistical reliability, ranks
the survivors, drift-gates the winner, and explains the decision — with the full trail logged to
MLflow.

The selection is deliberately **not** "lowest error wins". A model can score well historically and
still produce an unusable forward forecast, so accuracy is one input among several.

---

## How a run is decided

```
ingest → quality → preprocess → train → backtest → forward-validate
       → explain (SHAP) → rank → drift-gate → select → narrate → track
```

1. **Backtest** — rolling windows, refitting from scratch each fold. Produces MAPE, WMAPE, RMSE,
   MAE and SMAPE, as evidence rather than a verdict.
2. **Forward validation** — the 12-month forecast is screened by 10 rules (flat forecasts,
   excessive volatility, missing seasonality, unrealistic spikes…). All 10 run even after one
   fails, so a rejected model shows every reason at once.
3. **Ranking** — a weighted composite of backtest accuracy (0.50), forecast stability (0.25) and
   SHAP consistency (0.25), min-max normalised **within each key**.
4. **Drift gate** — the drift test *and* its threshold are chosen per key at runtime from that
   key's own sample size, cardinality and normality. Candidates are tried best-first until one
   passes.
5. **Fallback** — if none pass, a configured baseline (`seasonal_naive`) ships and is flagged as a
   fallback everywhere it appears.

Accuracy is defined as `100 − WMAPE`; the platform targets ≥70% per key given ≥24 months of
history.

### Models

`prophet` · `arima` · `xgboost` · `lightgbm` · `tft` · `seasonal_naive` (fallback only)

Registered in `forecast_engine/config/model_config.py`. Adding one is a registry entry plus an
adapter — no orchestration change. Each library is imported lazily, so a deployment that omits one
reports that model as unavailable instead of failing.

Tree models get generated lag, rolling-mean and calendar features and are fitted on first
differences. Without that a tree cannot extrapolate past its training range and every forecast
comes out flat — see `s05_models/base_model.py`.

---

## Repository layout

```
forecast_engine/     # The pipeline. Pure Python, no web/Spark dependency.
  config/            #   Frozen dataclass configs — every threshold and weight
  core/              #   PipelineContext, run state, live status writer
  s01…s12/           #   One package per stage, in execution order
backend/             # FastAPI
  app/api/           #   Routes — HTTP only, no forecasting logic
  app/auth/          #   Entra ID token validation + the RBAC table
  app/services/      #   Profiling, validation, estimation, results
  app/orchestration/ #   Pipeline Executor and its Local / Databricks runners
frontend/            # React (Vite) SPA
  src/auth/          #   MSAL sign-in, permission guards
  src/pages/         #   Wizard, deployments, results, MLflow experiments
databricks/          # Databricks Asset Bundle — job and cluster definitions
tests/               # backend/ runs on the backend venv, engine/ on the engine venv
docs/                # PHASE_A_AZURE_SETUP.md — the manual Azure steps
Dockerfile           # Dependencies-only image for Databricks Container Services
pyproject.toml       # Builds forecast_engine into a wheel for DAB
```

The Docker image holds **dependencies only**; the wheel holds **code only**. DAB joins them at run
time, so a code change never requires an image rebuild.

---

## The user flow

```
sign in (Entra ID) → upload → profile → map columns → configure
                   → estimate → run → track status → results → insights
```

The browser never talks to Databricks or to storage. It calls the API, the API stages the dataset
into the Unity Catalog volume and submits the existing `forecastiq-forecast-pipeline` job, and the
API reads status and results back. No Databricks token, storage key or Azure OpenAI key is ever
sent to the browser.

### Roles

| Role | Can do |
|---|---|
| `Admin` | everything, plus platform configuration |
| `DataScientist` | upload, configure, estimate, run, cancel, view results and MLflow internals |
| `Analyst` | view datasets, run history and results — cannot upload, run, cancel or inspect models |

Roles come from Entra ID **app roles** (or, as a fallback, a group→role map). The mapping from role
to permission lives in one table, `backend/app/auth/rbac.py`. The UI hides what a user cannot do;
the API enforces it independently, so a hidden control that is forced into view still returns 403.

A signed-in user with **no** role assigned is authenticated but authorized for nothing — deliberate,
so an unassigned tenant member does not inherit read access to results.

---

## Running locally

Requires Python 3.11+ and Node 18+.

```bash
# 1. Engine
python -m venv forecast_engine/.venv
forecast_engine/.venv/bin/pip install -r forecast_engine/requirements.txt

# 2. Backend
python -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
cp backend/.env.example backend/.env      # then fill it in — see Configuration
backend/.venv/bin/python -m uvicorn app.main:app --port 8000   # from backend/

# 3. Frontend
cd frontend && npm install && npm run dev  # http://localhost:5173
```

`AUTH_ENABLED` defaults to **false**, so this runs with no Azure tenant: the API issues a local
development identity whose role is `DEV_IDENTITY_ROLE` (set it to `Analyst` to exercise RBAC). The
API **refuses to start** with authentication disabled unless `APP_ENV` is development/local/test, so
this cannot ship as an unauthenticated deployment.

### Tests

```bash
backend/.venv/bin/python -m pytest tests/backend -q          # auth, RBAC, estimation, runner
forecast_engine/.venv/bin/python -m pytest tests/engine -q   # engine CLI contract
```

Or drive the engine directly, with no web layer:

```bash
forecast_engine/.venv/bin/python -m forecast_engine.run_pipeline \
  --dataset your_data.csv \
  --date-column date --target-column sales \
  --key-columns store item \
  --horizon 12 --summary-out summary.json
```

---

## Configuration

No credentials are committed. Every environment supplies its own; the `.env.example` files list
the names.

| File | Holds | Committed |
|---|---|---|
| `backend/.env` | Entra ID app ids, Databricks service principal, Azure OpenAI, MLflow, execution mode | no |
| `frontend/.env` | the API base URL, and nothing else | no |
| `databricks/.env` | CLI profile name, ACR image | no |
| Databricks secret scope `forecastiq` | ACR pull creds, Azure OpenAI (for jobs) | n/a |

The frontend holds **no** Azure configuration: it fetches its Entra client id, tenant and scope
from the API's `/auth/config` at runtime, so one built bundle deploys to any environment. The API
itself holds no Entra credential either — it validates tokens with the tenant's public signing
keys.

Azure-side setup that must be done by hand (app registrations, role assignments, the Databricks
service principal) is in **`docs/PHASE_A_AZURE_SETUP.md`**.

Databricks job YAML contains only `{{secrets/...}}` **references**, never values — which is why it
is safe to commit.

**Azure OpenAI is optional.** Leave its variables empty and the run completes normally, reporting
insights as unavailable. It is wired so a missing secret can never block a cluster from starting.

### Execution mode

`EXECUTION_MODE=local` (default) runs the engine as a subprocess on the API host.

`EXECUTION_MODE=databricks` submits the bundle-deployed job. The runner stages the dataset and a
JSON run configuration into one per-run directory in the existing Unity Catalog volume, calls the
Jobs API, polls run state, reads the engine's live stage trail back from the volume so the UI shows
real progress, and reads `summary.json` back on completion. Both backends return the identical
result envelope, so nothing above the executor knows which one ran.

Retargeting is this one setting changing — no code change.

---

## Deploying to Databricks

```bash
# One-time: authenticate a profile. The workspace URL lives here, not in git.
databricks auth login --host https://adb-xxxx.N.azuredatabricks.net --profile my-workspace

cd databricks
cp .env.example .env && source .env        # profile name + ACR image
databricks bundle validate -t dev
databricks bundle deploy  -t dev           # builds and uploads the wheel

databricks bundle run forecast_pipeline -t dev --params \
  dataset=/Volumes/<catalog>/<schema>/<volume>/your.csv,\
config=/Volumes/<catalog>/<schema>/<volume>/runs/manual/forecast_configuration.json
```

The job takes exactly four parameters — `dataset`, `config`, `summary_out`, `live_status_out` — and
every per-run value (columns, models, fallback, horizon, run id) travels inside the `config` JSON.
That is not a style choice: a `python_wheel_task`'s argument list is fixed at deploy time, so an
unset parameter would reach argparse as `--horizon ""` (a crash) or `--models ""` (a model named
""). A config file also carries multi-value key columns, which one flat parameter cannot.

`dataset` and `config` intentionally have **no defaults** — a default would let a forgotten
`--params` silently forecast the wrong file and report success. Normally the API writes both.

Requires a workspace with **Databricks Container Services enabled** and classic (non-serverless)
compute.

---

## Governance

Every run logs one MLflow parent run: parameters, per-key/per-model metrics, ranking components,
drift statistics and thresholds, the LLM narrative with its grounding payload, and the curated
dataset. It also produces plots reviewable directly in the MLflow UI — accuracy by key, per-key
model comparison, drift distribution overlays, a ranking-component heatmap and backtest trends.
The winning model per key is registered in Unity Catalog.
