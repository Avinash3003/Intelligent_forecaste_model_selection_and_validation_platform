from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str
    environment: str
    execution_backend: str = "local"
    # Whether run history (Deployments, Results, Experiments,
    # Observability for anything not still in this process's memory) can
    # actually be read back. False here is exactly the failure mode behind
    # "my past runs disappeared" — the backend keeps accepting and running
    # new jobs regardless, so nothing else about /health would ever catch
    # it. Only a reachability boolean, never the tracking URI itself: that
    # can carry embedded credentials and this route is unauthenticated.
    run_history_available: bool = True
    # "sqlite" | "databricks" | other scheme — enough to tell an operator
    # which store is being read, without exposing the full URI.
    run_history_backend: str = "sqlite"
