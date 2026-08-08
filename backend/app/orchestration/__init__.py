"""Pipeline Orchestration & Execution Layer (Section 6.14).

    Pipeline Executor
        -> Execution Runner
            -> Local Runner OR Databricks Runner
                -> Forecast Pipeline -> MLflow -> Results

This package is the *only* thing the rest of the FastAPI backend is allowed
to call to run a forecast. It never contains forecasting business logic
itself — training, evaluation, ranking, drift validation, explainability
and MLflow logging all remain exactly where the earlier phases put them,
inside `forecast_engine`. What lives here is purely the question of *how*
and *where* that engine gets invoked, and how its result is handed back in
one standardized shape regardless of which backend ran it.
"""
