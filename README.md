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
databricks/          # Databricks Asset Bundle — wheel artifact + a dev-only manual-test job
tests/               # backend/ runs on the backend venv, engine/ on the engine venv
docs/                # PHASE_A_AZURE_SETUP.md, execution-modes.md — manual Azure steps + execution architecture
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
backend/.venv/bin/python -m pytest tests/backend -q          # backend suite
forecast_engine/.venv/bin/python -m pytest tests/engine -q   # engine suite
backend/.venv/bin/python -m pytest tests/test_smoke.py -q    # smoke: app imports and serves
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

One forecast_engine, two execution modes.

`EXECUTION_MODE=local` (default) runs the engine as a subprocess on the API host.

`EXECUTION_MODE=databricks` runs it through the backend's own named Databricks job, a seven-task
DAG, on whichever compute the user selected in the Compute step — an existing
all-purpose cluster, or a new job cluster the backend creates for that run (optionally pulling a
Databricks Container Services image; see the root `Dockerfile`). No job resource is pre-deployed or
resolved by name for this path. The runner stages the dataset and a JSON run configuration into one
per-run directory in the existing Unity Catalog volume, calls the Jobs API, polls run state, reads
the engine's live stage trail back from the volume so the UI shows real progress, and reads
`summary.json` back on completion.

Every mode returns the identical result envelope, so nothing above the executor knows which one ran.
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
```

Deploying builds and uploads the `forecast_engine` wheel artifact — the application code every
cloud run installs at submission time via `libraries: [whl: ...]`. It creates no job resource for
`prod`: a real run is always started by the backend through its own named job, whose definition it
keeps current at request time.

The `dev` target additionally deploys `forecast_pipeline_compute`, a Ray key-parallel job kept for
manual/CLI testing against an existing all-purpose cluster:

```bash
databricks bundle run forecast_pipeline_compute -t dev --params \
  dataset=/Volumes/<catalog>/<schema>/<volume>/your.csv,\
config=/Volumes/<catalog>/<schema>/<volume>/runs/manual/forecast_configuration.json
```

Its parameters — `dataset`, `config`, `summary_out`, `live_status_out` — are fixed at deploy time
(a `python_wheel_task`'s argument list can't be extended per run), so every other per-run value
(columns, models, fallback, horizon) travels inside the `config` JSON instead. `dataset` and
`config` intentionally have **no defaults** — a default would let a forgotten `--params` silently
forecast the wrong file and report success. Normally the API writes both; this manual path exists
for testing the compute/Ray path directly, outside the application.

Production deployment is automated — a push to `main` runs the tested, validated `prod` target
through GitHub Actions rather than a manual `bundle deploy`.

Three workflows cover it: `ci.yml` (tests and build validation, also reused as the pre-deploy gate),
`deploy-databricks.yml` (the wheel artifact), and `deploy-app.yml` (backend
and frontend). `.gitlab-ci.yml` mirrors `ci.yml`'s checks for GitLab, which does not read
`.github/`; deployment stays on GitHub Actions alone so the two platforms cannot both deploy.
Every environment-specific value comes from an encrypted Actions secret, so no
resource name, hostname or identifier is committed.

---

## Governance

Every run logs one MLflow parent run: parameters, per-key/per-model metrics, ranking components,
drift statistics and thresholds, the LLM narrative with its grounding payload, and the curated
dataset. It also produces plots reviewable directly in the MLflow UI — accuracy by key, per-key
model comparison, drift distribution overlays, a ranking-component heatmap and backtest trends.
The winning model per key is registered in Unity Catalog.
