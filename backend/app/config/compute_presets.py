"""The compute sizes ForecastIQ offers, and the runtimes it can run on.

A short curated list rather than the workspace's full catalog: the engine
only runs on a machine-learning runtime, and reading every node type
Databricks offers costs seconds on a page that should render instantly.
Databricks is still the authority — validation checks the chosen values
against it before a run is allowed.
"""

from __future__ import annotations

from app.schemas.compute import NodeTypeOption, RuntimeOption

# Sizes that suit a key-parallel forecasting run: general purpose first,
# then more memory or more cores for wider datasets.
NODE_TYPE_PRESETS: tuple[NodeTypeOption, ...] = (
    NodeTypeOption(
        node_type_id="Standard_DC4as_v5",
        label="Standard — 4 vCPU, 16 GB",
        description="Balanced default, proven for ForecastIQ runs.",
        category="General Purpose",
        num_cores=4,
        memory_mb=16384,
    ),
    NodeTypeOption(
        node_type_id="Standard_F4ads_v7",
        label="Compute optimised — 4 vCPU, 16 GB",
        description="Faster cores for model-heavy runs.",
        category="Compute Optimized",
        num_cores=4,
        memory_mb=16384,
    ),
    NodeTypeOption(
        node_type_id="Standard_E4ads_v7",
        label="Memory optimised — 4 vCPU, 32 GB",
        description="Extra memory for datasets with many forecast keys.",
        category="Memory Optimized",
        num_cores=4,
        memory_mb=32768,
    ),
    NodeTypeOption(
        node_type_id="Standard_F8ads_v7",
        label="Large — 8 vCPU, 32 GB",
        description="More cores, so more keys run at once.",
        category="Compute Optimized",
        num_cores=8,
        memory_mb=32768,
    ),
)

# The engine needs Ray, xgboost, lightgbm and shap, which only the
# machine-learning runtimes carry.
RUNTIME_PRESETS: tuple[RuntimeOption, ...] = (
    RuntimeOption(key="15.4.x-cpu-ml-scala2.12", name="15.4 LTS ML"),
    RuntimeOption(key="16.4.x-cpu-ml-scala2.12", name="16.4 LTS ML"),
)

DEFAULT_NODE_TYPE_ID = NODE_TYPE_PRESETS[0].node_type_id
DEFAULT_RUNTIME_KEY = RUNTIME_PRESETS[0].key
