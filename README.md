# Intelligent Forecast Model Selection & Validation Platform

Enterprise Auto-ML time-series forecasting platform. Business users upload a dataset, define
metadata, select candidate models, and trigger a validated, explainable forecasting run executed
on Azure Databricks with full MLflow governance.

## Repository Structure

```
TECH_DEMO/
├── frontend/     # React (Vite) self-service SPA — dashboard, pipeline builder, explainability UI
├── backend/      # Orchestration / API layer (auth, job submission, results retrieval)
├── databricks/   # Databricks Asset Bundle (DAB) job/task definitions, pipeline code
├── docker/       # Databricks Container Services images, Dockerfiles
├── deployment/   # CI/CD, IaC (Azure resources), environment configs
├── docs/         # Architecture references, ADRs, design documents
└── README.md
```

## Status

**Phase 1 — Frontend Foundation** (current)
Sidebar/header/layout shell, routing, dummy-data-driven Dashboard and placeholder pages for
Forecast Pipeline, Deployments, Experiments, Models and Settings. No backend, auth, or business
logic yet.

## Getting Started (Frontend)

```bash
cd frontend
npm install
npm run dev
```

See [frontend/README.md](frontend/README.md) for frontend-specific details.
