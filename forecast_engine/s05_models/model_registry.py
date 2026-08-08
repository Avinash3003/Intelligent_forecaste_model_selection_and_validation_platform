"""Model Registry — the trainer's only route to a forecasting model.

The trainer never imports or constructs a model class directly. It asks the
registry for the models the user selected, and the registry resolves each
declaration to an adapter class by import path, checks that the backing
library is present, and constructs instances on demand.

That indirection buys three things the platform needs:

  * Onboarding a model is registering a ModelSpec (Section 8.4) — the
    training loop never changes.
  * Only selected models are loaded. Section 5.2 requires that no model
    outside the user's selection is trained or considered, and a model
    never selected is never even imported.
  * A missing optional dependency degrades to an Unavailable model instead
    of failing the run.
"""

from __future__ import annotations

import importlib
from typing import Any

from forecast_engine.config.model_config import ModelConfig, ModelSpec
from forecast_engine.s05_models.base_model import BaseForecastingModel
from forecast_engine.utils.exceptions import ModelRegistryError


class ModelRegistry:
    """Resolves model declarations into constructed model adapters."""

    # Store config and set up the per-process adapter cache
    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or ModelConfig.default()
        # Import paths are resolved once per process; a run trains thousands
        # of models and re-importing per group would dominate its runtime.
        self._adapter_cache: dict[str, type[BaseForecastingModel]] = {}

    # The model configuration this registry was built from
    @property
    def config(self) -> ModelConfig:
        return self._config

    # Return the declarations for the models this run should train
    def resolve_specs(self, selected: list[str] | None = None) -> tuple[ModelSpec, ...]:
        specs = self._config.enabled_models(selected)

        if selected:
            known = {spec.name.lower() for spec in self._config.registry}
            unknown = [name for name in selected if name.lower() not in known]
            if unknown:
                raise ModelRegistryError(
                    f"Unknown model(s) requested: {', '.join(unknown)}. "
                    f"Registered models are: {', '.join(spec.name for spec in self._config.registry)}."
                )

        return specs

    # Import and return the adapter class for spec
    def load_adapter(self, spec: ModelSpec) -> type[BaseForecastingModel]:
        if spec.name in self._adapter_cache:
            return self._adapter_cache[spec.name]

        module_path, _, class_name = spec.adapter.rpartition(".")
        if not module_path:
            raise ModelRegistryError(f"Model '{spec.name}' has a malformed adapter path '{spec.adapter}'.")

        try:
            module = importlib.import_module(module_path)
            adapter = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise ModelRegistryError(
                f"Could not load adapter '{spec.adapter}' for model '{spec.name}': {exc}"
            ) from exc

        if not issubclass(adapter, BaseForecastingModel):
            raise ModelRegistryError(
                f"Adapter '{spec.adapter}' for model '{spec.name}' does not implement "
                f"the BaseForecastingModel interface."
            )

        self._adapter_cache[spec.name] = adapter
        return adapter

    # Whether spec's backing library is installed
    def is_available(self, spec: ModelSpec) -> bool:
        try:
            return self.load_adapter(spec).is_available()
        except ModelRegistryError:
            return False

    # Construct a fresh model instance for one forecasting group
    def create(self, spec: ModelSpec, params: dict[str, Any] | None = None) -> BaseForecastingModel:
        return self.load_adapter(spec)(spec, params)

    # Summarize the selected models and their availability
    def describe(self, selected: list[str] | None = None) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "available": self.is_available(spec),
                "search_strategy": spec.search_strategy.value,
                "supports_features": spec.supports_features,
                "min_observations": spec.min_observations,
            }
            for spec in self.resolve_specs(selected)
        ]
