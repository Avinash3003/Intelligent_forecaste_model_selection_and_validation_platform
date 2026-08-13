"""LightGBM adapter.

Identical framing to the XGBoost adapter — both inherit the supervised
design-matrix construction — differing only in the estimator built and the
parameters supplied by configuration.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from forecast_engine.s05_models.base_model import SupervisedTreeModel


class LightGBMModel(SupervisedTreeModel):
    """Gradient-boosted trees via LightGBM."""

    # Whether lightgbm is importable
    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("lightgbm") is not None

    # Construct the LGBMRegressor for this model's parameters
    def build_estimator(self) -> Any:
        # Imported lazily so the library is only required when selected.
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**self._estimator_params())
