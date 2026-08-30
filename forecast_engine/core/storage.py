"""One way to reach run storage, whatever the run is executing on.

Unity Catalog External Volumes, backed by ADLS, are the single source of
truth for every persistent file a run produces. How a process *reaches*
them differs by execution mode, and that difference is the only thing this
module exists to hide:

    Existing Compute   /Volumes is a real FUSE mount  -> ordinary file I/O
    DCS container      /Volumes is not mounted        -> Databricks Files API
    anything else      an ordinary local path         -> ordinary file I/O

Proven on the production image (wheel-task run 130735570011315, DCS
15.4.x-scala2.12): `os.listdir("/Volumes")` raises
`PermissionError: [Errno 1] Operation not permitted`, while the Files API
lists, writes 40 bytes and reads them back byte-identical. So a container
has no *filesystem handler* for UC Volumes — it has never lacked *access*.

Everything goes through Unity Catalog either way. This module never uses a
storage account key, a SAS token or a direct `abfss://` URL: those would
reach the bytes while stepping around the grants that protect them, which
would make the volume ACLs decorative.

Callers pass a path and get bytes. They are not told, and must not ask,
which of the three routes served them.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Every Unity Catalog Volume path begins here, on every cloud.
VOLUME_PREFIX = "/Volumes/"

# Set to "1"/"0" to force the routing decision instead of probing for it.
# Exists for tests and for a deployment that knows better than the probe;
# unset (the normal case) means the probe decides.
POSIX_VOLUMES_ENV_VAR = "FORECASTIQ_POSIX_VOLUMES"


class StorageError(RuntimeError):
    """Storage could not be reached. Never raised for "the file is absent"."""


def is_volume_path(path: object) -> bool:
    """Whether `path` names a Unity Catalog Volume."""
    return str(path).startswith(VOLUME_PREFIX)


_posix_lock = threading.Lock()
_posix_available: bool | None = None


def posix_volumes_available() -> bool:
    """Whether this process can reach /Volumes as an ordinary filesystem.

    Probed once and remembered: the answer is a property of the runtime,
    which cannot change while the process lives, and the probe is a syscall
    on a path every stage touches.

    `os.listdir` rather than `os.path.exists`, deliberately. In the DCS
    container `/Volumes` *exists* — it is the listing that fails with
    `[Errno 1] Operation not permitted`, because the directory is there but
    no `uc-volumes` handler is mounted behind it. Testing for existence
    would report the mount as working and send every write to a path that
    silently goes nowhere.
    """
    global _posix_available

    forced = os.environ.get(POSIX_VOLUMES_ENV_VAR)
    if forced is not None:
        return forced.strip() not in ("", "0", "false", "False")

    with _posix_lock:
        if _posix_available is None:
            try:
                os.listdir(VOLUME_PREFIX.rstrip("/"))
                _posix_available = True
            except OSError as exc:
                logger.info(
                    "UC Volumes are not mounted in this runtime (%s); using the Files API.",
                    exc,
                )
                _posix_available = False
        return _posix_available


def reset_route_cache() -> None:
    """Forget the probe result. For tests; never needed in a real run."""
    global _posix_available
    with _posix_lock:
        _posix_available = None


# ----------------------------------------------------------------------
# The Files API route
# ----------------------------------------------------------------------

_client_lock = threading.Lock()
_client: object | None = None


def _files_client() -> object:
    """A Files API client authenticated as the job's own identity.

    `WorkspaceClient()` on its own does NOT work here and must never be
    used: a python_wheel_task has no ambient credential chain, so the SDK's
    default resolution fails every call with "default auth: cannot
    configure default credentials" — confirmed on the production image in
    both a notebook task and a wheel task.

    The credentials do exist, just not where that chain looks. The runtime
    publishes them through the driver's notebook context, which is the same
    mechanism `core/databricks_secrets.py` already uses in this very
    process to read Azure OpenAI secrets. Imported locally for the same
    reason it is there: `databricks.sdk.runtime` builds a live `dbutils` at
    import time and fails anywhere that is not real Databricks compute.
    """
    global _client
    with _client_lock:
        if _client is not None:
            return _client
        try:
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.config import Config
            from databricks.sdk.runtime import dbutils

            context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
            host = context.apiUrl().get()
            token = context.apiToken().get()
            _client = WorkspaceClient(config=Config(host=host, token=token, auth_type="pat"))
        except Exception as exc:  # noqa: BLE001 - any failure here is fatal for storage
            # Deliberately fatal. The alternative — falling back to the local
            # filesystem — would write a run's outputs to disposable driver
            # storage and report success, which is the failure mode this
            # whole architecture exists to remove.
            raise StorageError(
                "Cannot reach Unity Catalog storage: this runtime has no /Volumes mount "
                f"and the Databricks Files API could not be authenticated ({type(exc).__name__}: {exc})."
            ) from exc
        return _client


def reset_client_cache() -> None:
    """Forget the Files API client. For tests."""
    global _client
    with _client_lock:
        _client = None


def set_files_client(client: object | None) -> None:
    """Inject a Files API client. For tests, and for a caller that already
    holds an authenticated client and should not build a second one."""
    global _client
    with _client_lock:
        _client = client


# ----------------------------------------------------------------------
# The interface every caller uses
# ----------------------------------------------------------------------


def _use_files_api(path: object) -> bool:
    return is_volume_path(path) and not posix_volumes_available()


def supports_atomic_replace(path: object) -> bool:
    """Whether this path can be replaced by a same-filesystem rename.

    True only on the POSIX route. Callers that need an all-or-nothing
    update ask this rather than asking which compute they are on — the
    execution mode stays inside this module, which is the whole point of it.

    False does not mean "no guarantee". A Files API write of a payload under
    files_ext_multipart_upload_min_stream_size (50 MiB) is one request, and
    an object store replaces an object wholesale, so a reader sees the old
    document or the new one and never a splice. It means only that
    `os.replace` is not the mechanism available.
    """
    return not _use_files_api(path)


def read_bytes(path: object) -> bytes:
    """The file's contents. Raises FileNotFoundError if it is not there."""
    if _use_files_api(path):
        try:
            return _files_client().files.download(str(path)).contents.read()
        except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
            raise _translate(exc, path) from exc
    return Path(path).read_bytes()


def write_bytes(path: object, payload: bytes) -> None:
    """Write `payload`, creating parent directories as needed."""
    if _use_files_api(path):
        try:
            _files_client().files.upload(str(path), io.BytesIO(payload), overwrite=True)
            return
        except Exception as exc:  # noqa: BLE001
            raise _translate(exc, path) from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def open_binary(path: object):
    """The file as a binary stream, for readers that stream (pandas).

    On the POSIX route this is the real file handle, so a large dataset is
    never copied into memory just to be handed to a parser. On the Files
    API route the bytes have to be fetched before they can be parsed, so
    the stream wraps them — there is no partial-read primitive to stream
    from, and pretending otherwise would only hide the fetch.

    Always use as a context manager; the POSIX handle must be closed.
    """
    if _use_files_api(path):
        return io.BytesIO(read_bytes(path))
    return open(Path(path), "rb")


def read_text(path: object, encoding: str = "utf-8") -> str:
    return read_bytes(path).decode(encoding)


def write_text(path: object, text: str, encoding: str = "utf-8") -> None:
    write_bytes(path, text.encode(encoding))


def exists(path: object) -> bool:
    """Whether the file is there.

    Returns False only for a file that is genuinely absent. Anything else —
    a credential that will not authenticate, a volume the caller has no
    grant on, the API being unreachable — raises.

    The distinction is the whole point. `dataset_loader` asks this question
    to decide whether to report "Dataset could not be found", so folding an
    outage into False tells a user their data is missing when in fact the
    platform could not be reached. Storage that is broken must say so.
    """
    if _use_files_api(path):
        try:
            _files_client().files.get_metadata(str(path))
            return True
        except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
            translated = _translate(exc, path)
            if isinstance(translated, FileNotFoundError):
                return False
            raise translated from exc
    return Path(path).exists()


def ensure_parent(path: object) -> None:
    """Make sure the file's directory exists.

    A no-op on the Files API route: `files.upload` creates the parent path
    itself, and there is no directory to make in object storage.
    """
    if _use_files_api(path):
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def list_dir(path: object) -> list[str]:
    """Full paths of the entries directly under `path`."""
    if _use_files_api(path):
        try:
            return [entry.path for entry in _files_client().files.list_directory_contents(str(path))]
        except Exception as exc:  # noqa: BLE001
            raise _translate(exc, path) from exc
    return [str(child) for child in sorted(Path(path).iterdir())]


def _translate(exc: Exception, path: object) -> Exception:
    """A missing file stays a FileNotFoundError; anything else is a
    StorageError naming the path, so a failure is never mistaken for
    "the run produced nothing"."""
    if type(exc).__name__ in ("NotFound", "ResourceDoesNotExist"):
        return FileNotFoundError(str(path))
    return StorageError(f"{path}: {type(exc).__name__}: {exc}")
