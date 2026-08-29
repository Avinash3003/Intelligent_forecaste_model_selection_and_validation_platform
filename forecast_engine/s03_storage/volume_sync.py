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

    @property
    def ok(self) -> bool:
        return not self.errors

    def describe(self) -> str:
        parts = [f"{self.files_copied} file(s), {self.bytes_copied:,} bytes copied to the volume"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} source(s) not produced by this run")
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
            # Default authentication: inside a Databricks job the SDK reads
            # the run's own credentials from the environment, the same way
            # the MLflow client in this process already does.
            from databricks.sdk import WorkspaceClient

            self._client = WorkspaceClient()
        return self._client

    def run(self, targets: list[SyncTarget]) -> SyncOutcome:
        outcome = SyncOutcome()
        for target in targets:
            if not target.source.exists():
                outcome.skipped.append(str(target.source))
                continue
            if target.source.is_file():
                self._copy_file(target.source, target.destination, outcome)
                continue
            for path in sorted(p for p in target.source.rglob("*") if p.is_file()):
                relative = path.relative_to(target.source).as_posix()
                self._copy_file(path, f"{target.destination.rstrip('/')}/{relative}", outcome)
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
