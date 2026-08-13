"""Thin facade around the MLflow SDK.

Every other module in `forecast_engine/tracking/` — and every other module
in the engine — reaches MLflow only through this class. That single choke
point is what makes two things true elsewhere in the codebase:

  * Retargeting Local Development to Databricks is exactly one value —
    `MLflowConfig.tracking_uri` — because nothing outside this file ever
    constructs a tracking URI or imports `mlflow` itself.
  * A future SDK change (a renamed argument, a new registry API) is a
    one-file change, never a hunt through the pipeline.

The `mlflow` package is imported lazily, mirroring the platform's existing
optional-dependency pattern (SHAP, Prophet, TFT): a deployment that
disables tracking is never forced to have it installed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from forecast_engine.config.mlflow_config import MLflowConfig
from forecast_engine.utils.exceptions import MLflowTrackingError

if TYPE_CHECKING:
    from mlflow import ActiveRun

# MLflow restricts metric/param/tag keys to alphanumerics, underscores,
# dashes, periods, spaces, colons and slashes. Forecast group ids use " | "
# as a separator (Section 6.2, `GroupingConfig.key_separator`), which is
# outside that set, so every key generated anywhere in this package is
# passed through this before being sent to MLflow.
_UNSAFE_KEY_CHARS = re.compile(r"[^a-zA-Z0-9_\-. /:]")

# Registered model names are held to a stricter charset than metric/param
# keys: Unity Catalog's three-level naming (a future registry target, per
# Section 6.13) disallows spaces and slashes that plain MLflow keys accept.
_UNSAFE_MODEL_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_\-]")


# Sanitize a string for use as an MLflow key
def sanitize_key(raw: str) -> str:
    return _UNSAFE_KEY_CHARS.sub("_", raw)


# Sanitize a string for use as a registered model name
def sanitize_model_name(raw: str) -> str:
    return _UNSAFE_MODEL_NAME_CHARS.sub("_", raw)


class MLflowClient:
    """Facade over the operations the tracking pipeline needs."""

    def __init__(self, config: MLflowConfig) -> None:
        self._config = config
        self._configured = False

    def is_available(self) -> bool:
        """Whether tracking can be attempted: enabled and the `mlflow`
        package importable. Never raises.
        """
        if not self._config.enabled:
            return False
        try:
            import mlflow  # noqa: F401
        except ImportError:
            return False
        return True

    def unavailable_reason(self) -> str:
        if not self._config.enabled:
            return "MLflow tracking disabled by configuration."
        try:
            import mlflow  # noqa: F401
        except ImportError:
            return "The 'mlflow' package is not installed."
        return ""

    def configure(self) -> None:
        """Point the SDK at the configured tracking/registry URIs and
        ensure the target experiment exists. Idempotent — safe to call
        once per run.

        Raises:
            MLflowTrackingError: if the tracking server cannot be reached
                or the experiment cannot be created/resolved.
        """
        mlflow = self._sdk()
        try:
            if self._config.tracking_uri:
                mlflow.set_tracking_uri(self._config.tracking_uri)
            if self._config.registry_uri:
                mlflow.set_registry_uri(self._config.registry_uri)

            experiment = mlflow.get_experiment_by_name(self._config.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    self._config.experiment_name, artifact_location=self._config.artifact_location
                )
            mlflow.set_experiment(self._config.experiment_name)
            self._configured = True
        except Exception as exc:  # noqa: BLE001 - every SDK failure becomes one business exception
            raise MLflowTrackingError(
                f"Could not prepare MLflow experiment '{self._config.experiment_name}': {exc}"
            ) from exc

    def start_run(
        self, run_name: str, tags: dict[str, str] | None = None, resume_run_id: str | None = None
    ) -> "ActiveRun":
        """Open the Parent Run for one forecasting pipeline execution.

        Returns the SDK's `ActiveRun`. Unlike a `with` block, the run is
        left *open* deliberately: the pipeline opens it before its first
        stage so that a failure part-way through is still recorded against
        a real MLflow run, and closes it with an explicit terminal status
        via `end_run()`. Child Runs (a future extension, e.g. one per
        forecasting group) nest inside it via `mlflow.start_run(nested=True)`.

        Args:
            resume_run_id: When set, reopens this existing run instead of
                creating a new one — how a later Databricks task in the
                multi-task workflow (a separate process, with no run active
                in its own fluent MLflow state) continues logging into the
                same Parent Run the first task opened. `run_name`/`tags`
                are not reapplied in this case; they were already set when
                the run was first opened.
        """
        if not self._configured:
            self.configure()
        mlflow = self._sdk()
        try:
            if resume_run_id:
                return mlflow.start_run(run_id=resume_run_id)
            return mlflow.start_run(run_name=run_name, tags={sanitize_key(k): str(v) for k, v in (tags or {}).items()})
        except Exception as exc:  # noqa: BLE001
            raise MLflowTrackingError(f"Could not start MLflow run '{run_name}': {exc}") from exc

    def active_run_info(self) -> dict[str, str]:
        """Identifiers of the currently open run, or an empty mapping."""
        try:
            active = self._sdk().active_run()
        except Exception:  # noqa: BLE001
            return {}
        if active is None:
            return {}
        return {"run_id": active.info.run_id, "experiment_id": active.info.experiment_id}

    def end_run(self, status: str) -> None:
        """Close the currently open run with an explicit terminal status.

        `status` is an MLflow RunStatus name — "FINISHED", "FAILED" or
        "KILLED". Never raises: this runs in the failure path of a
        pipeline that has already failed, and a tracking problem must not
        replace the original error the caller is about to propagate.
        """
        try:
            self._sdk().end_run(status=status)
        except Exception:  # noqa: BLE001 - see docstring
            pass

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a batch of run parameters, sanitizing keys and stringifying
        values (MLflow parameters are always strings).
        """
        mlflow = self._sdk()
        safe = {sanitize_key(str(key)): _stringify(value) for key, value in params.items() if value is not None}
        if not safe:
            return
        try:
            mlflow.log_params(safe)
        except Exception as exc:  # noqa: BLE001
            raise MLflowTrackingError(f"Failed to log parameters: {exc}") from exc

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log a batch of numeric run metrics, sanitizing keys."""
        mlflow = self._sdk()
        safe = {sanitize_key(str(key)): float(value) for key, value in metrics.items() if value is not None}
        if not safe:
            return
        try:
            mlflow.log_metrics(safe)
        except Exception as exc:  # noqa: BLE001
            raise MLflowTrackingError(f"Failed to log metrics: {exc}") from exc

    def set_tags(self, tags: dict[str, str]) -> None:
        """Log categorical, non-numeric information (e.g. a status string)
        that has no place in `log_metrics`, which is numeric-only.
        """
        mlflow = self._sdk()
        safe = {sanitize_key(str(key)): _stringify(value) for key, value in tags.items() if value is not None}
        if not safe:
            return
        try:
            mlflow.set_tags(safe)
        except Exception as exc:  # noqa: BLE001
            raise MLflowTrackingError(f"Failed to set tags: {exc}") from exc

    def log_dict_artifact(self, data: Any, artifact_file: str) -> None:
        """Log `data` as a JSON artifact at `artifact_file` (may include
        subdirectories, e.g. "shap/group_1.json").
        """
        mlflow = self._sdk()
        try:
            mlflow.log_dict(data, artifact_file)
        except Exception as exc:  # noqa: BLE001
            raise MLflowTrackingError(f"Failed to log artifact '{artifact_file}': {exc}") from exc

    def log_text_artifact(self, text: str, artifact_file: str) -> None:
        """Log a plain-text/markdown artifact — MLflow's UI previews these
        directly, which JSON does not get the same treatment for.
        """
        mlflow = self._sdk()
        try:
            mlflow.log_text(text, artifact_file)
        except Exception as exc:  # noqa: BLE001
            raise MLflowTrackingError(f"Failed to log text artifact '{artifact_file}': {exc}") from exc

    def log_artifact_file(self, local_path: str, artifact_path: str | None = None) -> None:
        """Copy an existing local file (e.g. the curated dataset already
        written to disk) into the run's artifact store.
        """
        mlflow = self._sdk()
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as exc:  # noqa: BLE001
            raise MLflowTrackingError(f"Failed to log artifact file '{local_path}': {exc}") from exc

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Log a matplotlib figure directly, without an intermediate file."""
        mlflow = self._sdk()
        try:
            mlflow.log_figure(figure, artifact_file)
        except Exception as exc:  # noqa: BLE001
            raise MLflowTrackingError(f"Failed to log figure '{artifact_file}': {exc}") from exc

    def register_pyfunc_model(
        self,
        python_model: Any,
        artifact_path: str,
        registered_model_name: str,
        signature: Any = None,
    ) -> Any:
        """Log a `mlflow.pyfunc.PythonModel` and register it in one call.

        Returns the SDK's `ModelInfo`, from which the caller reads the
        assigned registered version. A single generic pyfunc flavour is
        used for every model family (statistical, tree-based, or the
        seasonal-naive fallback) deliberately — one wrapper avoids a
        five-way branch of near-identical native-flavour logging code.

        Registry metadata is *not* passed here. `log_model` grew a `tags`
        argument only in MLflow 3.x, and 2.x's signature has no `**kwargs`
        to absorb it, so supplying it raises `TypeError` on every 2.x
        deployment — which is what the pinned runtime resolves to. Tags are
        applied through `set_model_version_tags` instead, whose API is
        identical across both, so one code path serves every supported
        version. See that method.

        `signature` is accepted (built by the caller via
        `mlflow.models.infer_signature`, not here — this facade never
        inspects a model's own data) because Unity Catalog rejects any
        registration missing one: "Model passed for registration did not
        contain any signature metadata." Optional rather than required so
        a caller that logs without registering is unaffected.
        """
        mlflow = self._sdk()
        try:
            return mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=python_model,
                registered_model_name=registered_model_name,
                signature=signature,
            )
        except Exception as exc:  # noqa: BLE001
            raise MLflowTrackingError(
                f"Failed to register model '{registered_model_name}': {exc}"
            ) from exc

    def set_model_version_tags(self, name: str, version: str, tags: dict[str, str]) -> None:
        """Attach registry metadata to one registered model version.

        The counterpart to `register_pyfunc_model`: the tags land on the
        version that call produced, which is where the Model Registry
        surfaces them. `log_model` waits for the new version to reach a
        terminal state before returning (`await_registration_for`), so the
        version referenced here already exists by the time this runs.

        Raises:
            MLflowTrackingError: if the tags cannot be applied. The caller
                decides what that means — a registration that succeeded but
                could not be annotated is still a registration.
        """
        safe = {sanitize_key(str(key)): _stringify(value) for key, value in tags.items() if value is not None}
        if not safe:
            return
        try:
            client = self._sdk().tracking.MlflowClient()
            for key, value in safe.items():
                client.set_model_version_tag(name=name, version=str(version), key=key, value=value)
        except Exception as exc:  # noqa: BLE001
            raise MLflowTrackingError(
                f"Failed to tag version {version} of model '{name}': {exc}"
            ) from exc

    def _sdk(self) -> Any:
        try:
            import mlflow
        except ImportError as exc:
            raise MLflowTrackingError(
                "The 'mlflow' package is not installed; run `pip install mlflow` to enable "
                "experiment tracking."
            ) from exc
        return mlflow


def _stringify(value: Any) -> str:
    return value if isinstance(value, str) else str(value)
