"""Copies a container run's outputs into a Unity Catalog Volume.

A Databricks Container Services image carries the engine's own dependency
set, and in exchange it gives up the runtime's UC Volumes FUSE mount. In
the container `/Volumes` exists but cannot be read or written —

    PermissionError: [Errno 1] Operation not permitted: '/Volumes'
    mount.err: "Unrecognized storage scheme: uc-volumes"

— so a container run writes its outputs to workspace files, which it can.

That is a limitation of the *mount*, not of access: the Files API reaches
the very same Volume over REST, from inside the container as readily as
from anywhere else. This module carries the outputs across that way before
the run exits, which is what finally lands them in the storage account
behind the catalog. Nothing else in the engine knows a Volume exists — the
writers keep writing ordinary files to ordinary paths.

The caller supplies every source and destination (the backend resolves
both from one set of path helpers), so the engine holds no opinion about
where a deployment keeps its data.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# A Volume write can fail transiently — a token refresh, a control-plane
# blip — and losing a finished run's results to one of those is a much
# worse outcome than waiting a moment and asking again.
_UPLOAD_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0


@dataclass(frozen=True)
class SyncTarget:
    """One location to copy. A source that does not exist is skipped.

    Sources are skipped rather than failed because not every run produces
    every output: a run with insights disabled writes no artifacts mirror,
    and demanding one would fail a perfectly good run.
    """

    source: Path
    destination: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SyncTarget":
        return cls(source=Path(str(payload["source"])), destination=str(payload["destination"]))


@dataclass
class SyncOutcome:
    files_copied: int = 0
    bytes_copied: int = 0
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Set when the sweep stopped because the credentials themselves are
    # wrong, rather than because one file failed. Reported separately: the
    # operator action is completely different.
    aborted_on_auth: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors

    def describe(self) -> str:
        parts = [f"{self.files_copied} file(s), {self.bytes_copied:,} bytes copied to the volume"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} source(s) not produced by this run")
        if self.aborted_on_auth:
            parts.append("ABORTED — the run could not authenticate to the Files API")
        if self.errors:
            parts.append(f"{len(self.errors)} failed: " + "; ".join(self.errors))
        return ". ".join(parts) + "."


class VolumeSync:
    """Uploads files to UC Volume paths through the Databricks Files API."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = _runtime_client()
        return self._client

    def run(self, targets: list[SyncTarget]) -> SyncOutcome:
        outcome = SyncOutcome()
        for target in targets:
            if not target.source.exists():
                outcome.skipped.append(str(target.source))
                continue
            if target.source.is_file():
                self._copy_file(target.source, target.destination, outcome)
            else:
                for path in sorted(p for p in target.source.rglob("*") if p.is_file()):
                    relative = path.relative_to(target.source).as_posix()
                    self._copy_file(path, f"{target.destination.rstrip('/')}/{relative}", outcome)
                    if outcome.aborted_on_auth:
                        break
            if outcome.aborted_on_auth:
                logger.error(
                    "Volume sync aborted: the run cannot authenticate to the Files API. "
                    "%d file(s) copied before the failure.",
                    outcome.files_copied,
                )
                break
        return outcome

    def _copy_file(self, source: Path, destination: str, outcome: SyncOutcome) -> None:
        try:
            payload = source.read_bytes()
        except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
            outcome.errors.append(f"{source}: {type(exc).__name__}: {exc}")
            return

        last: Exception | None = None
        for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
            try:
                self.client.files.upload(destination, io.BytesIO(payload), overwrite=True)
                outcome.files_copied += 1
                outcome.bytes_copied += len(payload)
                return
            except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
                last = exc
                # A credential problem is the same on every file and on every
                # attempt. Retrying it turns one broken run into a retry storm:
                # the first outage of this cost 76 files x 3 attempts x
                # backoff, roughly seven minutes of sleeping, and buried the
                # real cause in 228 identical warnings. Recorded once, and the
                # sweep stops.
                if _is_auth_failure(exc):
                    outcome.errors.append(f"{destination}: {type(exc).__name__}: {exc}")
                    outcome.aborted_on_auth = True
                    return
                logger.warning(
                    "Volume upload attempt %d/%d failed for %s: %s",
                    attempt,
                    _UPLOAD_ATTEMPTS,
                    destination,
                    exc,
                )
                if attempt < _UPLOAD_ATTEMPTS:
                    time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

        outcome.errors.append(f"{destination}: {type(last).__name__}: {last}")


def _is_auth_failure(exc: Exception) -> bool:
    """Whether `exc` means "these credentials are wrong" rather than "this
    one call failed". The SDK reports it as a message, not a typed error."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "default auth",
            "cannot configure default credentials",
            "invalid access token",
            "authentication",
            "unauthorized",
            "permission denied",
        )
    )


def _runtime_client() -> Any:
    """A Files API client authenticated as the job's own identity.

    `WorkspaceClient()` on its own does NOT work here and must not be used:
    a python_wheel_task has no ambient credential chain, so the SDK's
    default resolution fails with "default auth: cannot configure default
    credentials" on every call. That is what broke the first version of
    this sync — the workspace files were written, and not one byte reached
    the storage account.

    The credentials do exist, just not where the default chain looks. The
    Databricks runtime exposes them through the notebook context on the
    driver, which is the same mechanism `core/databricks_secrets.py`
    already uses successfully in this very process to read Azure OpenAI
    secrets. Imported locally for the same reason it is there:
    `databricks.sdk.runtime` builds a live `dbutils` at import time and
    fails anywhere that is not real Databricks compute.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.config import Config
    from databricks.sdk.runtime import dbutils

    context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    host = context.apiUrl().get()
    token = context.apiToken().get()
    return WorkspaceClient(config=Config(host=host, token=token, auth_type="pat"))


def sync_outputs_to_volume(config_payload: dict[str, Any] | None, client: Any | None = None) -> SyncOutcome | None:
    """Run the copy described by the config's `volume_sync` block.

    Returns None when the block is absent, which is every run that already
    writes straight to its final storage — a local run, and any Databricks
    run that is not executing inside a container image.
    """
    block = (config_payload or {}).get("volume_sync")
    if not block:
        return None

    targets = [SyncTarget.from_payload(entry) for entry in block.get("targets", [])]
    if not targets:
        return None

    return VolumeSync(client).run(targets)
