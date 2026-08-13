"""MLflow Experiment Tracking & Model Registry orchestration — Section
6.13 end to end.

    Open Parent Run -> Pipeline Execution -> Log Parameters -> Log
    Metrics -> Log Artifacts -> Register Winner Model -> Close Run

One MLflow Parent Run per forecasting pipeline execution. Every value
logged is read from a single `PipelineResult` — this class never touches
`PipelineContext`, a live model, or a DataFrame, which is what keeps it
usable unchanged against a Databricks tracking server: only
`MLflowConfig.tracking_uri` differs between environments.

The run is opened by `begin()` *before the first pipeline stage* and
closed by either `complete()` or `fail()`. That ordering is what makes
MLflow a usable system of record: a run that dies in preprocessing still
leaves a real MLflow run marked FAILED, carrying the reason. Closing at
the end only — logging everything in one shot once the pipeline had
already succeeded — meant failed runs were never recorded at all.

Nothing here can fail the forecasting run it is recording. Every failure
mode — an unreachable tracking server, a bad experiment name, a
registration conflict — is caught and reflected on the returned
`TrackingResult` instead of raised.
"""

from __future__ import annotations

import time

from forecast_engine.config.mlflow_config import MLflowConfig
from forecast_engine.core.pipeline_result import PipelineResult
from forecast_engine.s12_tracking.artifact_logger import log_all_artifacts
from forecast_engine.s12_tracking.metric_logger import build_metrics, build_status_tags
from forecast_engine.s12_tracking.mlflow_client import MLflowClient
from forecast_engine.s12_tracking.model_registrar import register_winner_models
from forecast_engine.s12_tracking.parameter_logger import build_hyperparameters, build_run_parameters
from forecast_engine.s12_tracking.run_summary import (
    DATASET_NAME_TAG,
    ERROR_TAG,
    FAILED_STAGE_TAG,
    RUN_ID_TAG,
    STARTED_BY_DISPLAY_NAME_TAG,
    STARTED_BY_USER_ID_TAG,
    SUMMARY_ARTIFACT_PATH,
)
from forecast_engine.s12_tracking.tracking_report import TrackingResult
from forecast_engine.utils.exceptions import MLflowTrackingError


class MLflowTrackingPipeline:
    """Logs one finished pipeline execution to MLflow as a single Parent
    Run, then registers the Final Production Model for every group that
    has one.
    """

    # Wire up MLflow config and client
    def __init__(self, config: MLflowConfig | None = None, client: MLflowClient | None = None) -> None:
        self._config = config or MLflowConfig.default()
        self._client = client or MLflowClient(self._config)
        self._open = False
        self._started_at: float | None = None

    # Open the Parent Run before the first pipeline stage; never raises
    def begin(
        self,
        run_id: str,
        dataset_name: str | None = None,
        started_by_user_id: str | None = None,
        started_by_display_name: str | None = None,
    ) -> TrackingResult:
        # Returns a TrackingResult describing the *open* run (or why one
        # could not be opened). The pipeline keeps this only so a failure
        # before `complete()` still has something to report.
        self._started_at = time.perf_counter()
        result = self._base_result()

        if not self._config.enabled:
            result.status = "disabled"
            return result
        if not self._client.is_available():
            result.status = f"unavailable: {self._client.unavailable_reason()}"
            return result

        result.attempted = True
        try:
            self._client.configure()
            # `run_id` and `dataset_name` are set as tags at open time, not
            # at close: they are the two fields the run-history view needs
            # to identify a run, and a run that later fails must still
            # carry them. Run *status* is deliberately not tagged — MLflow's
            # own `RunInfo.status` already carries it, and duplicating it
            # here would create a second copy that can go stale.
            active_run = self._client.start_run(
                run_name=f"forecast-run-{run_id}",
                tags={
                    RUN_ID_TAG: run_id,
                    DATASET_NAME_TAG: dataset_name or "",
                    STARTED_BY_USER_ID_TAG: started_by_user_id or "",
                    STARTED_BY_DISPLAY_NAME_TAG: started_by_display_name or "",
                },
            )
            result.run_id = active_run.info.run_id
            result.experiment_id = active_run.info.experiment_id
            self._open = True
            result.status = "running"
        except Exception as exc:  # noqa: BLE001 - tracking must never crash the pipeline
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
        return result

    # Reopen an already-open run in a NEW process — the multi-task
    # Databricks workflow's later tasks, none of which called begin().
    def resume(self, mlflow_run_id: str | None) -> None:
        """Make `track()`/`fail()` usable again in a fresh process.

        `begin()` opens the Parent Run once, in the first task. Every
        later task is a separate process with no run active in its own
        MLflow fluent state, so `self._open` starts `False` there too —
        without this, `track()`/`fail()` would report "no run was open"
        even though one genuinely is, on the tracking server.

        No-op (mirroring `begin()`'s own graceful degradation) when
        `mlflow_run_id` is falsy — tracking was disabled/unavailable when
        the first task called `begin()`, so there is nothing to resume —
        or when the reopen itself fails; either way `track()`/`fail()`
        already treat "not open" as a reportable status, never a crash.
        """
        if not mlflow_run_id or not self._config.enabled or not self._client.is_available():
            return
        try:
            self._client.configure()
            self._client.start_run(run_name="", resume_run_id=mlflow_run_id)
            self._open = True
        except Exception:  # noqa: BLE001 - tracking must never crash the pipeline
            pass

    # Log a finished execution into the open run and close it FINISHED; never raises
    def track(self, pipeline_result: PipelineResult, summary: dict | None = None) -> TrackingResult:
        result = self._base_result()

        if not self._config.enabled:
            result.status = "disabled"
            return result
        if not self._client.is_available():
            result.status = f"unavailable: {self._client.unavailable_reason()}"
            return result
        if not self._open:
            # begin() never got a run open (unreachable server, bad
            # experiment). Reporting that is more honest than silently
            # logging into whatever run MLflow considers active.
            result.status = "failed"
            result.error = "No MLflow run was open; tracking could not start at pipeline begin."
            return result

        result.attempted = True
        try:
            active = self._client.active_run_info()
            result.run_id = active.get("run_id")
            result.experiment_id = active.get("experiment_id")

            parameters = build_run_parameters(pipeline_result)
            parameters.update(build_hyperparameters(pipeline_result, self._config))
            self._client.log_params(parameters)
            result.parameters_logged = len(parameters)

            metrics = build_metrics(pipeline_result)
            self._client.log_metrics(metrics)
            result.metrics_logged = len(metrics)

            self._client.set_tags(build_status_tags(pipeline_result))

            if self._config.log_artifacts:
                result.artifact_errors = log_all_artifacts(self._client, pipeline_result, self._config)

            result.registrations = register_winner_models(
                self._client, pipeline_result, self._config, result.run_id
            )

            # Settled before the summary is logged, not after: `summary`
            # was built by the caller from a context whose tracking_result
            # was still `begin()`'s "running" snapshot — this object is the
            # only place that knows the real, near-final outcome. Logging
            # `summary` unpatched would permanently embed "running" in the
            # one artifact a restored run is read back from.
            result.logged = True
            result.status = "logged_with_artifact_errors" if result.artifact_errors else "logged"

            # The consolidated run record, logged last so it can include
            # everything above. This is the single artifact the backend
            # reads to rebuild a past run — see `run_summary.py`.
            if summary is not None:
                self._client.log_dict_artifact({**summary, "tracking_result": result.to_dict()}, SUMMARY_ARTIFACT_PATH)

            self._close("FINISHED")
        except MLflowTrackingError as exc:
            result.status = "failed"
            result.error = str(exc)
            self._close("FAILED")
        except Exception as exc:  # noqa: BLE001 - tracking must never crash the forecasting pipeline
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            self._close("FAILED")

        result.duration_seconds = self._elapsed()
        return result

    # Close the open run as FAILED, recording why the pipeline died; never raises
    def fail(self, run_id: str, error: Exception, failed_stage: str | None = None) -> TrackingResult:
        result = self._base_result()
        result.status = "pipeline_failed"
        result.error = f"{type(error).__name__}: {error}"

        if not self._open:
            return result

        try:
            active = self._client.active_run_info()
            result.run_id = active.get("run_id")
            result.experiment_id = active.get("experiment_id")
            # Recorded as tags rather than an artifact: a failed run is
            # looked up by *why* it failed, and tags are searchable in
            # MLflow where artifacts are not.
            self._client.set_tags(
                {
                    ERROR_TAG: result.error[:4000],
                    FAILED_STAGE_TAG: failed_stage or "unknown",
                }
            )
        except Exception:  # noqa: BLE001 - already on the failure path
            pass

        self._close("FAILED")
        result.duration_seconds = self._elapsed()
        return result

    # A TrackingResult pre-filled with what static config already knows
    def _base_result(self) -> TrackingResult:
        # Set unconditionally so the dashboard's MLflow panel has a
        # human-readable name even when tracking is
        # disabled/unavailable/failed, not just on success.
        return TrackingResult(
            tracking_uri=self._config.tracking_uri,
            experiment_name=self._config.experiment_name,
        )

    def _close(self, status: str) -> None:
        if self._open:
            self._client.end_run(status)
            self._open = False

    def _elapsed(self) -> float:
        return time.perf_counter() - self._started_at if self._started_at is not None else 0.0
