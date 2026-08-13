"""XGBoost adapter.

Frames the series as a supervised regression problem (see
SupervisedTreeModel) and fits an XGBRegressor. All hyperparameters come
from the model registry configuration; nothing is hard-coded here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from forecast_engine.s05_models.base_model import SupervisedTreeModel


class XGBoostModel(SupervisedTreeModel):
    """Gradient-boosted trees via XGBoost."""

    # Whether xgboost is importable
    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("xgboost") is not None

    # Construct the XGBRegressor for this model's parameters
    def build_estimator(self) -> Any:
        # Imported lazily so a deployment that never selects XGBoost is not
        # required to install it.
        from xgboost import XGBRegressor

        return XGBRegressor(**self._estimator_params())
