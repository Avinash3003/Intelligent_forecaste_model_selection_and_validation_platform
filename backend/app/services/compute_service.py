"""Compute options for the wizard, and real validation against Databricks.

The step renders from project presets, so opening it costs no Databricks
call. Databricks is consulted when the user validates a configuration:
first the live workspace catalog, then — only when that cannot settle it —
a real create probe that is always cleaned up.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from app.config.compute_presets import (
    DEFAULT_NODE_TYPE_ID,
    DEFAULT_RUNTIME_KEY,
    NODE_TYPE_PRESETS,
    RUNTIME_PRESETS,
)
from app.config.model_availability import container_runtime_required
from app.config.settings import Settings, get_settings
from app.orchestration.databricks_runner import RUN_CLUSTER_TAG, DatabricksRunner
from app.schemas.compute import (
    ComputeOptions,
    ComputeValidationResult,
    ExistingCompute,
    ExistingComputeListResponse,
    ExistingComputeValidationResult,
    JobComputeConfig,
)

try:
    from databricks.sdk.service.compute import DataSecurityMode
except ImportError:  # the SDK is optional until Databricks execution is used
    DataSecurityMode = None

logger = logging.getLogger(__name__)

PROBE_NAME_PREFIX = "forecastiq-validation"
PROBE_TAG = {"forecastiq_purpose": "compute-validation"}
PROBE_AUTOTERMINATION_MINUTES = 10
PROBE_TIMEOUT_SECONDS = 420
PROBE_POLL_SECONDS = 10

# Databricks termination codes -> one short sentence for the user.
TERMINATION_MESSAGES = {
    "AZURE_QUOTA_EXCEEDED_EXCEPTION": "Unable to create compute: insufficient vCPU quota.",
    "AZURE_RESOURCE_QUOTA_EXCEEDED": "Unable to create compute: insufficient vCPU quota.",
    "CLOUD_PROVIDER_LAUNCH_FAILURE": "The cloud provider could not launch this machine type.",
    "CLOUD_PROVIDER_RESOURCE_STOCKOUT": "This machine type is not available for your subscription right now.",
    "AZURE_VM_EXTENSION_FAILURE": "Unable to create compute: the machine failed to start correctly.",
    "INVALID_ARGUMENT": "The selected configuration is not supported by this workspace.",
    "UNSUPPORTED_INSTANCE_TYPE": "This machine type is not available for your subscription.",
    "INIT_SCRIPT_FAILURE": "Unable to create compute: start-up failed.",
    "USER_REQUEST": "Validation stopped before Databricks could confirm this configuration.",
}

# Substrings in a Databricks API error -> the same short sentences.
API_ERROR_MESSAGES = (
    ("quota", "Unable to create compute: insufficient vCPU quota."),
    ("permission", "You do not have permission to create this compute."),
    ("not authorized", "You do not have permission to create this compute."),
    ("policy", "This configuration is restricted by a workspace compute policy."),
    ("node type", "This machine type is not available for your subscription."),
    ("instance type", "This machine type is not available for your subscription."),
    ("spark version", "The selected runtime is not available in this workspace."),
    ("access mode", "The selected configuration is not supported by this workspace."),
    ("isolation", "The selected configuration is not supported by this workspace."),
)

GENERIC_FAILURE = "Unable to create compute. Please review the selected settings."

# States an existing cluster can be used from, either directly or by
# Databricks starting it when the run is submitted.
USABLE_STATES = {"RUNNING", "PENDING", "RESIZING", "RESTARTING"}
STARTABLE_STATES = {"TERMINATED"}
# Attaching is the minimum; restarting is also needed when it is stopped.
ATTACH_PERMISSIONS = {"CAN_ATTACH_TO", "CAN_RESTART", "CAN_MANAGE"}
RESTART_PERMISSIONS = {"CAN_RESTART", "CAN_MANAGE"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cluster_id_of(created: Any) -> str | None:
    for candidate in (getattr(created, "response", None), created):
        cluster_id = getattr(candidate, "cluster_id", None)
        if isinstance(cluster_id, str):
            return cluster_id
    bind = getattr(created, "bind", None)
    return bind.get("cluster_id") if isinstance(bind, dict) else None


def _to_existing_compute(cluster_id: str, cluster: Any, node: dict | None = None) -> ExistingCompute:
    """One cluster as the picker shows it, entirely from live workspace data.

    `node` is this cluster's row from the workspace's node-type catalog. It
    supplies per-node cores/memory/GPU, which the cluster itself only
    reports while RUNNING — without it a stopped cluster shows no capacity.
    """
    num_workers = int(getattr(cluster, "num_workers", 0) or 0)
    state = getattr(cluster, "state", None)
    nodes = num_workers + 1  # workers plus the driver
    catalog = node or {}
    per_node_cores = catalog.get("num_cores")
    per_node_memory = catalog.get("memory_mb")

    # Live totals win while the cluster is up; the catalog fills them in
    # when it is not.
    live_cores = int(getattr(cluster, "cluster_cores", 0) or 0)
    live_memory = int(getattr(cluster, "cluster_memory_mb", 0) or 0)

    return ExistingCompute(
        cluster_id=cluster_id,
        cluster_name=getattr(cluster, "cluster_name", cluster_id),
        state=getattr(state, "value", None) or (str(state) if state else None),
        node_type_id=getattr(cluster, "node_type_id", None),
        runtime=getattr(cluster, "spark_version", None),
        num_workers=num_workers,
        num_cores=live_cores or (int(per_node_cores * nodes) if per_node_cores else None),
        memory_mb=live_memory or (int(per_node_memory * nodes) if per_node_memory else None),
        autotermination_minutes=getattr(cluster, "autotermination_minutes", None),
        single_node=num_workers == 0,
        node_category=catalog.get("category"),
        num_gpus=(int(catalog["num_gpus"] * nodes) if catalog.get("num_gpus") else None),
        driver_node_type_id=getattr(cluster, "driver_node_type_id", None),
        creator=getattr(cluster, "creator_user_name", None),
        data_security_mode=_enum_text(getattr(cluster, "data_security_mode", None)),
    )


# An SDK enum or plain string, as plain text.
def _enum_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


# All-purpose vs. job compute is a real Databricks distinction
# (`cluster_source`), not one this platform invents: a JOB/PIPELINE-sourced
# cluster is ephemeral infrastructure a job created for itself, never
# something a *different* run should be pointed at. Our own validation
# probes (compute_service's create-probe clusters) are excluded the same
# way the orphan sweep already recognizes them, by name prefix, since a
# probe is created with `cluster_source` left at its default (API/UI) —
# nothing in the source enum marks it as ours.
#
# A cluster this app creates for one run's new_job_compute (see
# databricks_runner._create_shared_cluster) also reports ClusterSource.UI —
# jobs.submit() has no job-cluster concept, so it is made via the plain
# Clusters API — so it is excluded by its RUN_CLUSTER_TAG instead of source.
_NON_ALL_PURPOSE_SOURCES = {"JOB", "PIPELINE", "PIPELINE_MAINTENANCE"}


def _is_all_purpose(cluster: Any) -> bool:
    name = getattr(cluster, "cluster_name", "") or ""
    if name.startswith(PROBE_NAME_PREFIX):
        return False
    if RUN_CLUSTER_TAG in (getattr(cluster, "custom_tags", None) or {}):
        return False
    source = getattr(cluster, "cluster_source", None)
    source_value = str(getattr(source, "value", source) or "").upper()
    return source_value not in _NON_ALL_PURPOSE_SOURCES


def _existing_lookup_message(error: Exception) -> str:
    text = str(error).lower()
    if "does not exist" in text or "not found" in text or "invalid_parameter_value" in text:
        return "Existing compute could not be found."
    if "permission" in text or "not authorized" in text or "forbidden" in text:
        return "Existing compute is not accessible with the current application permissions."
    return "Existing compute could not be reached right now."


# Why this cluster cannot run the engine, or None when it can. Runs off
# fields clusters.list() already returns, so listing every candidate costs
# no extra per-cluster call. GPU vs CPU is never checked here — the engine
# needs an ML runtime's dependencies (torch, xgboost, prophet, ...), not a
# GPU; a GPU cluster with an ML runtime is compatible, a CPU one without
# one is not.
def _incompatibility(cluster: Any, caller: str | None) -> str | None:
    single_user = getattr(cluster, "single_user_name", None)
    if single_user and caller and single_user != caller:
        return "Existing compute is reserved for another user."

    runtime = str(getattr(cluster, "spark_version", "") or "").lower()
    if not getattr(cluster, "use_ml_runtime", False) and "ml" not in runtime:
        return "Existing compute does not use a machine learning runtime, which this forecast requires."
    return None


# Whether `cluster` can be offered at all: compatible AND in a state a run
# could actually attach to (not a per-cluster call — same list() fields).
def _listable(cluster: Any, caller: str | None) -> str | None:
    reason = _incompatibility(cluster, caller)
    if reason:
        return reason

    state = str(getattr(getattr(cluster, "state", None), "value", "") or "").upper()
    if state in USABLE_STATES:
        return None
    if state in STARTABLE_STATES:
        return _termination_failure(cluster)
    return f"Existing compute is in state {state or 'UNKNOWN'}."


# A terminated cluster's reason, when it represents a real failure.
def _termination_failure(cluster: Any) -> str | None:
    reason = getattr(cluster, "termination_reason", None)
    if reason is None:
        return None
    kind = str(getattr(getattr(reason, "type", None), "value", "") or "").upper()
    if kind in {"", "SUCCESS"}:
        return None
    code = str(getattr(getattr(reason, "code", None), "value", "") or "") or None
    return message_for_termination(code)


def message_for_api_error(error: Exception) -> str:
    text = str(error).lower()
    for needle, message in API_ERROR_MESSAGES:
        if needle in text:
            return message
    return GENERIC_FAILURE


def message_for_termination(code: str | None, parameters: dict | None = None) -> str:
    if code and code in TERMINATION_MESSAGES:
        return TERMINATION_MESSAGES[code]
    blob = f"{code or ''} {parameters or ''}".lower()
    for needle, message in API_ERROR_MESSAGES:
        if needle in blob:
            return message
    return GENERIC_FAILURE


class ComputeService:
    """Compute presets, the existing-compute fallback, and real validation."""

    # One probe at a time, so repeated clicks cannot stack up clusters.
    _probe_lock = threading.Lock()

    def __init__(self, settings: Settings | None = None, workspace: Any = None) -> None:
        self._settings = settings or get_settings()
        self._workspace = workspace
        self._node_catalog: dict[str, dict] | None = None
        # Guards the catalog fetch so concurrent callers share one — see
        # _ensure_catalog. Per instance, unlike the class-level probe lock
        # above, which exists to keep one probe cluster alive at a time
        # across every instance.
        self._catalog_lock = threading.Lock()

    # Reuse the runner's authenticated client rather than building another.
    def _client(self) -> Any:
        if self._workspace is None:
            self._workspace = DatabricksRunner(self._settings)._workspace
        return self._workspace

    # ---- options (no Databricks call) ---------------------------------

    def get_options(self) -> ComputeOptions:
        return ComputeOptions(
            node_types=list(NODE_TYPE_PRESETS),
            runtimes=list(RUNTIME_PRESETS),
            default_node_type_id=DEFAULT_NODE_TYPE_ID,
            default_runtime_key=DEFAULT_RUNTIME_KEY,
        )

    # ---- existing compute (one lookup) --------------------------------

    def list_existing_compute(self) -> ExistingComputeListResponse:
        # ForecastIQ's one supported runtime is the prebuilt container
        # image; a plain cluster cannot load it. Reported through the same
        # available/message shape the picker already renders for "no
        # clusters available" or "unreachable", so a user sees why before
        # choosing it rather than after submitting and being refused.
        if container_runtime_required(self._settings):
            return ExistingComputeListResponse(
                available=False,
                message=(
                    "ForecastIQ runs on its prebuilt container runtime, which carries every "
                    "model's dependencies. Existing Compute cannot load that image, so it is "
                    "kept only as legacy infrastructure. Use New Job Compute to run a pipeline."
                ),
            )

        # One call lists every cluster in the workspace, all-purpose and
        # job alike, with the exact fields _to_existing_compute already
        # reads off a single cluster.get() — so offering every all-purpose
        # cluster instead of one hardcoded id costs nothing extra: the same
        # one round trip either way, never a lookup per cluster.
        try:
            clusters = list(self._client().clusters.list())
        except Exception as exc:  # noqa: BLE001 - a missing fallback is not fatal
            logger.warning("Could not list existing compute: %s", exc)
            return ExistingComputeListResponse(
                available=False, message="Existing compute could not be reached right now."
            )

        caller = self._current_user_name()
        options = []
        for cluster in clusters:
            cluster_id = getattr(cluster, "cluster_id", None)
            if not cluster_id or not _is_all_purpose(cluster):
                continue
            reason = _listable(cluster, caller)
            if reason:
                logger.info("Excluding cluster %s from Existing Compute: %s", cluster_id, reason)
                continue
            # One cached catalog fetch for the whole list, not one per
            # cluster — see _ensure_catalog.
            options.append(
                _to_existing_compute(cluster_id, cluster, self._node_info(getattr(cluster, "node_type_id", "")))
            )

        if not options:
            return ExistingComputeListResponse(
                available=False, message="No compatible all-purpose compute is available in this workspace."
            )
        return ExistingComputeListResponse(available=True, clusters=options)

    # ---- existing compute validation ----------------------------------

    def validate_existing_compute(self, cluster_id: str | None) -> ExistingComputeValidationResult:
        """Check the selected cluster can actually run this workload.

        Reads the cluster and its permissions only — nothing is created,
        started, resized or modified. `cluster_id` is whichever cluster the
        user picked from list_existing_compute(); there is no longer a
        single configured fallback to validate in its absence.
        """
        cluster_id = (cluster_id or "").strip()
        if not cluster_id:
            return self._existing_invalid("No existing compute was selected.")

        try:
            cluster = self._client().clusters.get(cluster_id=cluster_id)
        except Exception as exc:  # noqa: BLE001 - a lookup failure is a real answer
            logger.warning("Existing compute %s could not be read: %s", cluster_id, exc)
            return self._existing_invalid(_existing_lookup_message(exc))

        state = str(getattr(getattr(cluster, "state", None), "value", "") or "").upper()

        compatibility = _incompatibility(cluster, self._current_user_name())
        if compatibility:
            return self._existing_invalid(compatibility, state=state)

        permissions = self._permissions_for(cluster_id)
        if permissions is not None and not (permissions & ATTACH_PERMISSIONS):
            return self._existing_invalid(
                "Existing compute is not accessible with the current application permissions.",
                state=state,
            )

        if state in USABLE_STATES:
            return self._existing_valid("Existing compute is available and ready to run.", state)

        if state in STARTABLE_STATES:
            failure = _termination_failure(cluster)
            if failure:
                return self._existing_invalid(failure, state=state)
            if permissions is not None and not (permissions & RESTART_PERMISSIONS):
                return self._existing_invalid(
                    "Existing compute is stopped and cannot be started with the current permissions.",
                    state=state,
                )
            return self._existing_valid(
                "Existing compute is currently stopped but will start for this run.",
                state,
                starts_on_demand=True,
            )

        if state in {"TERMINATING", "ERROR"}:
            return self._existing_invalid(
                "Existing compute is not available right now. Please try again shortly.", state=state
            )
        return self._existing_invalid("Existing compute is in an unknown state.", state=state)

    # The caller's own permission levels on this cluster, or None if the
    # workspace does not report them.
    def _permissions_for(self, cluster_id: str) -> set[str] | None:
        user = self._current_user_name()
        try:
            acl = self._client().clusters.get_permissions(cluster_id=cluster_id)
        except Exception as exc:  # noqa: BLE001 - absence of an ACL is not a failure
            logger.warning("Could not read permissions for %s: %s", cluster_id, exc)
            return None

        levels: set[str] = set()
        for entry in getattr(acl, "access_control_list", None) or []:
            principal = (
                getattr(entry, "user_name", None)
                or getattr(entry, "service_principal_name", None)
            )
            group = getattr(entry, "group_name", None)
            if principal != user and group not in {"admins", "users"}:
                continue
            for permission in getattr(entry, "all_permissions", None) or []:
                level = getattr(permission, "permission_level", None)
                levels.add(str(getattr(level, "value", level) or ""))
        return levels or None

    def _existing_valid(
        self, message: str, state: str, starts_on_demand: bool = False
    ) -> ExistingComputeValidationResult:
        return ExistingComputeValidationResult(
            valid=True,
            message=message,
            state=state,
            starts_on_demand=starts_on_demand,
            checked_at=_now(),
        )

    def _existing_invalid(self, message: str, state: str | None = None) -> ExistingComputeValidationResult:
        return ExistingComputeValidationResult(
            valid=False, message=message, state=state, checked_at=_now()
        )

    # ---- new job compute validation ------------------------------------

    def validate(self, config: JobComputeConfig, quick: bool = False) -> ComputeValidationResult:
        catalog = self._validate_against_catalog(config)
        if not catalog.valid or quick:
            return catalog
        return self._validate_by_create_probe(config)

    def prewarm(self) -> None:
        """Load the node catalog before anyone waits on it.

        Every validation answers from this one catalog, in well under a
        millisecond — but the fetch itself measured 5.13s against this
        workspace, and whoever validates first in a fresh process pays it.
        Started at application startup that is nobody. Returns immediately;
        a failure is logged by the fetch and retried by the next caller.
        """
        threading.Thread(
            target=self._ensure_catalog,
            name="forecastiq-node-catalog",
            daemon=True,
        ).start()

    # The workspace's node catalog, read once per service instance.
    def _ensure_catalog(self) -> dict | None:
        # Single-flight. Without the lock, a request arriving while
        # startup's own warm-up is still in flight just runs a second fetch
        # of all 337 node types alongside it — and still waits the full
        # 5.13s for its own copy. Holding the lock means it joins the fetch
        # already running and returns the moment that one lands.
        with self._catalog_lock:
            if self._node_catalog is None:
                try:
                    raw = self._client().api_client.do("GET", "/api/2.1/clusters/list-node-types")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not read Databricks node types: %s", exc)
                    return None
                self._node_catalog = {n["node_type_id"]: n for n in raw.get("node_types", []) or []}
            return self._node_catalog

    def _node_info(self, node_type_id: str) -> dict | None:
        catalog = self._ensure_catalog()
        return catalog.get(node_type_id) if catalog else None

    def _catalog_core_quota(self) -> float | None:
        """The subscription's core ceiling, read off the node catalog.

        Azure enforces a Total Regional Cores limit on top of any per-family
        one, so it binds nearly every node type — and that is exactly what
        the catalog shows. This workspace, across 337 node types:

            4.0    -> 253 types    the regional limit
            6.0    ->   2 types    Standard_NC12/NC24, a GPU family quota
            0.0    ->  20 types    families not enabled at all
            absent ->  62 types    no figure reported — the silent case

        So the *modal* figure is the regional limit, and the run that
        prompted this confirms it exactly: "Total Regional Cores ...
        Current Limit: 4".

        The mode is used rather than the maximum, which the two GPU
        outliers would win — reporting a ceiling of 6 that no ordinary
        family has, and telling the user a number that is simply untrue —
        and rather than the minimum, which the disabled families would win
        at 0, rejecting every configuration. Zeros are excluded outright: a
        zero says "this family is unavailable", which the status check above
        already reports, and says nothing about the regional ceiling.

        This costs nothing at request time — the catalog is one cached call
        per service instance — which is the point. Actually creating the
        cluster is the only fully authoritative answer and it is far too
        slow to sit in a wizard: the run that prompted this took 191.5s to
        report ADD_NODES_FAILED and 276.1s to terminate.

        None when nothing in the catalog reports a usable quota, which
        leaves the caller no worse off than not asking.
        """
        quotas = [
            quota
            for node in (self._node_catalog or {}).values()
            if (quota := (node.get("node_info") or {}).get("available_core_quota"))
        ]
        if not quotas:
            return None
        return max(set(quotas), key=quotas.count)

    # Stage 1 — live workspace metadata for the selected values only.
    def _validate_against_catalog(self, config: JobComputeConfig) -> ComputeValidationResult:
        node = self._node_info(config.node_type_id)
        if node is None:
            return self._invalid("This machine type is not offered by your workspace.")

        statuses = [str(status) for status in ((node.get("node_info") or {}).get("status") or [])]
        if statuses:
            joined = " ".join(statuses).replace("_", "").lower()
            reason = (
                "not enabled on this subscription"
                if "notenabled" in joined
                else "not available in this workspace"
            )
            return self._invalid(f"This machine type is {reason}.")

        quota = (node.get("node_info") or {}).get("available_core_quota")
        if quota is None:
            # Not every node type carries a quota. This workspace reports
            # one for Standard_F4ads_v7, Standard_E4ads_v7 and
            # Standard_F8ads_v7 and none at all for Standard_DC4as_v5 —
            # which is the default preset, so the size most users pick was
            # the one size this check silently skipped. A three-worker
            # DC4as_v5 job (16 vCPUs against a limit of 4) validated clean
            # and then died at cluster start with AZURE_QUOTA_EXCEEDED,
            # after the user had already waited for compute.
            quota = self._catalog_core_quota()

        cores = int(node.get("num_cores") or 0)
        if quota is not None and cores:
            requested = config.requested_cores(cores)
            if requested > quota:
                return self._invalid(
                    f"Unable to create compute: this configuration needs {requested} vCPUs "
                    f"but only {int(quota)} are available."
                )

        return ComputeValidationResult(
            valid=True,
            stage="metadata",
            message="Machine type and size match your workspace's available capacity.",
            checked_at=_now(),
        )

    # Stage 2 — create the real thing, then always delete it.
    def _validate_by_create_probe(self, config: JobComputeConfig) -> ComputeValidationResult:
        if not self._probe_lock.acquire(blocking=False):
            return self._invalid("Another validation is already running. Please try again in a moment.")

        cluster_id = None
        try:
            clusters = self._client().clusters
            try:
                created = clusters.create(**self._probe_spec(config))
                cluster_id = _cluster_id_of(created)
            except Exception as exc:  # noqa: BLE001 - a rejection is a real answer
                logger.warning("Compute validation create failed: %s", exc)
                return self._invalid(message_for_api_error(exc), stage="create_probe")
            if not cluster_id:
                return self._invalid(GENERIC_FAILURE, stage="create_probe")
            return self._await_probe(clusters, cluster_id)
        finally:
            # Sweep by name too: a probe whose id could not be read must
            # never be left running.
            if cluster_id:
                self._delete_probe(cluster_id)
            self._delete_orphaned_probes()
            self._probe_lock.release()

    def _probe_spec(self, config: JobComputeConfig) -> dict:
        spec: dict = {
            "cluster_name": f"{PROBE_NAME_PREFIX}-{int(time.time())}",
            "spark_version": config.runtime_key,
            "node_type_id": config.node_type_id,
            "autotermination_minutes": PROBE_AUTOTERMINATION_MINUTES,
            "custom_tags": dict(PROBE_TAG),
            # Workspaces that disallow no-isolation clusters reject a spec
            # with no access mode.
            "data_security_mode": DataSecurityMode.SINGLE_USER if DataSecurityMode else "SINGLE_USER",
        }
        owner = self._current_user_name()
        if owner:
            spec["single_user_name"] = owner

        if config.autoscale:
            spec["autoscale"] = {"min_workers": config.min_workers, "max_workers": config.max_workers}
            return spec

        spec["num_workers"] = config.num_workers
        if config.num_workers == 0:
            spec["spark_conf"] = {
                "spark.databricks.cluster.profile": "singleNode",
                "spark.master": "local[*]",
            }
            spec["custom_tags"]["ResourceClass"] = "SingleNode"
        return spec

    # Poll until Databricks actually settles the cluster's fate.
    def _await_probe(self, clusters: Any, cluster_id: str) -> ComputeValidationResult:
        deadline = time.time() + PROBE_TIMEOUT_SECONDS
        while time.time() < deadline:
            try:
                cluster = clusters.get(cluster_id=cluster_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Compute validation poll failed: %s", exc)
                return self._invalid(message_for_api_error(exc), stage="create_probe")

            state = str(getattr(cluster, "state", "") or "").upper()
            if "RUNNING" in state:
                return ComputeValidationResult(
                    valid=True,
                    stage="create_probe",
                    message="Configuration is valid and ready to run.",
                    checked_at=_now(),
                )
            if "TERMINATED" in state or "ERROR" in state:
                reason = getattr(cluster, "termination_reason", None)
                code = str(getattr(reason, "code", "") or "") or None
                logger.warning("Compute validation terminated: code=%s", code)
                return self._invalid(
                    message_for_termination(code, getattr(reason, "parameters", None)),
                    stage="create_probe",
                )
            time.sleep(PROBE_POLL_SECONDS)

        return self._invalid(
            "Databricks could not confirm this configuration in time.", stage="create_probe"
        )

    def _current_user_name(self) -> str | None:
        try:
            return self._client().current_user.me().user_name
        except Exception as exc:  # noqa: BLE001 - the probe can still be attempted
            logger.warning("Could not resolve current user for compute validation: %s", exc)
            return None

    def _delete_orphaned_probes(self) -> None:
        try:
            clusters = list(self._client().clusters.list())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not sweep compute validation clusters: %s", exc)
            return
        for cluster in clusters:
            name = getattr(cluster, "cluster_name", "") or ""
            cluster_id = getattr(cluster, "cluster_id", None)
            if name.startswith(PROBE_NAME_PREFIX) and cluster_id:
                logger.warning("Deleting orphaned compute validation cluster %s", cluster_id)
                self._delete_probe(cluster_id)

    def _delete_probe(self, cluster_id: str) -> bool:
        try:
            self._client().clusters.permanent_delete(cluster_id=cluster_id)
            return True
        except Exception as exc:  # noqa: BLE001 - cleanup failure is logged, not raised
            logger.error("Could not delete compute validation cluster %s: %s", cluster_id, exc)
            return False

    def _invalid(self, message: str, stage: str = "metadata") -> ComputeValidationResult:
        return ComputeValidationResult(valid=False, stage=stage, message=message, checked_at=_now())
