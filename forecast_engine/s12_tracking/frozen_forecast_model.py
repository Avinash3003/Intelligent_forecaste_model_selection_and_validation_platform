"""The pyfunc wrapper a registered model version serves.

Its own module so cloudpickle can serialise it BY VALUE. Registered by
reference instead, every model artifact would carry a hidden dependency on
the forecast_engine wheel — which is not on public PyPI, so MLflow both
warned about it and produced a version that could only be loaded where that
exact internal wheel was installed. By value the artifact is self-contained
and needs only mlflow and pandas.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import mlflow.pyfunc

    PYFUNC_AVAILABLE = True
except ImportError:
    PYFUNC_AVAILABLE = False


if PYFUNC_AVAILABLE:

    class FrozenForecastModel(mlflow.pyfunc.PythonModel):
        """Serves one forecasting group's already-computed forward forecast.

        A record of "this is what the platform forecast for this group in
        this run", not a retrainable estimator.
        """

        # Store the frozen forecast for this group
        def __init__(self, group_id: str, model_name: str, forecast: dict[str, Any]) -> None:
            self.group_id = group_id
            self.model_name = model_name
            self.forecast = forecast

        # Untyped on purpose: a hint opts into MLflow's own schema convention.
        # Serve the frozen forecast values as a DataFrame
        def predict(self, context, model_input=None, params=None):
            values = self.forecast.get("values", [])
            return pd.DataFrame(
                {
                    "date": self.forecast.get("dates", []),
                    "value": values,
                    "lower": self.forecast.get("lower") or [None] * len(values),
                    "upper": self.forecast.get("upper") or [None] * len(values),
                }
            )

else:  # pragma: no cover - exercised only where mlflow is absent

    class FrozenForecastModel:  # type: ignore[no-redef]
        pass
