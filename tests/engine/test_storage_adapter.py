"""One storage interface, three routes, no silent fallback.

The architecture this pins was settled by measurement, not preference
(wheel-task run 130735570011315 on the production DCS image):

    os.listdir("/Volumes")  -> PermissionError [Errno 1] Operation not permitted
    Files API list/write/read-back -> PASS, 40 bytes, identical

So a DCS container has no filesystem handler for UC Volumes but full
access to them over the API, while Existing Compute has the mount. Callers
must not know which of those they got — that knowledge leaking into the
writers is what produced `if DCS:` branches scattered through the pipeline
before.

The rule that matters most here is the last one: when a Volume path cannot
be reached, storage FAILS. It never quietly writes to local disk, because
a run that reports success while its outputs sit on a disposable driver is
the exact failure this architecture removes.
"""

from __future__ import annotations

import io

import pytest

from forecast_engine.core import storage


@pytest.fixture(autouse=True)
def _clean_routing(monkeypatch):
    storage.reset_route_cache()
    storage.reset_client_cache()
    monkeypatch.delenv(storage.POSIX_VOLUMES_ENV_VAR, raising=False)
    # No test should pay the real backoff delay; retry *counting* is what
    # every test here cares about, never wall-clock time.
    monkeypatch.setattr(storage, "_TRANSIENT_RETRY_BACKOFF_SECONDS", 0)
    yield
    storage.reset_route_cache()
    storage.reset_client_cache()


class _FakeFiles:
    def __init__(self):
        self.store: dict[str, bytes] = {}
        self.uploads: list[str] = []
        self.downloads: list[str] = []

    def upload(self, path, contents, overwrite=False):
        self.uploads.append(path)
        self.store[path] = contents.read()

    def download(self, path):
        self.downloads.append(path)
        if path not in self.store:
            raise _NotFound(path)
        return type("R", (), {"contents": io.BytesIO(self.store[path])})()

    def get_metadata(self, path):
        if path not in self.store:
            raise _NotFound(path)
        return {}

    def list_directory_contents(self, path):
        prefix = path.rstrip("/") + "/"
        return [type("E", (), {"path": p})() for p in sorted(self.store) if p.startswith(prefix)]

    def delete(self, path):
        self.store.pop(path, None)


class _NotFound(Exception):
    pass


_NotFound.__name__ = "NotFound"


class _FakeClient:
    def __init__(self):
        self.files = _FakeFiles()


VOL = "/Volumes/forecastiq/forecasting/upload_files/runs/r1/summary.json"


# --- path classification ----------------------------------------------


def test_a_unity_catalog_path_is_recognised():
    assert storage.is_volume_path(VOL)
    assert storage.is_volume_path("/Volumes/c/s/v/f")


def test_an_ordinary_path_is_not_a_volume_path(tmp_path):
    assert not storage.is_volume_path(tmp_path / "x.csv")
    assert not storage.is_volume_path("/tmp/x")
    assert not storage.is_volume_path("/Workspace/forecastiq/runs/x")


# --- Existing Compute: POSIX route -------------------------------------


def test_with_the_mount_present_a_volume_path_uses_ordinary_file_io(monkeypatch, tmp_path):
    """Existing Compute must behave exactly as it always has."""
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "1")
    client = _FakeClient()
    storage.set_files_client(client)

    target = tmp_path / "nested" / "out.json"
    storage.write_bytes(target, b"{}")

    assert target.read_bytes() == b"{}"
    assert client.files.uploads == [], "the API must not be used when the mount works"


def test_a_local_path_never_uses_the_api_even_without_a_mount(monkeypatch, tmp_path):
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    client = _FakeClient()
    storage.set_files_client(client)

    target = tmp_path / "local.txt"
    storage.write_text(target, "hello")

    assert storage.read_text(target) == "hello"
    assert client.files.uploads == []


# --- DCS: Files API route ----------------------------------------------


def test_without_the_mount_a_volume_path_uses_the_files_api(monkeypatch):
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    client = _FakeClient()
    storage.set_files_client(client)

    storage.write_bytes(VOL, b'{"run":"r1"}')

    assert client.files.uploads == [VOL]
    assert storage.read_bytes(VOL) == b'{"run":"r1"}'


def test_the_round_trip_preserves_bytes_exactly(monkeypatch):
    """Model pickles go through here; a text-mode slip would corrupt them."""
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    storage.set_files_client(_FakeClient())
    payload = b"\x80\x04\x95binary\x00\xff model"

    storage.write_bytes(VOL, payload)

    assert storage.read_bytes(VOL) == payload


def test_listing_and_existence_route_the_same_way(monkeypatch):
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    storage.set_files_client(_FakeClient())
    base = "/Volumes/forecastiq/forecasting/models_files/runs/r1"

    storage.write_bytes(f"{base}/a.pkl", b"a")
    storage.write_bytes(f"{base}/b.pkl", b"b")

    assert storage.exists(f"{base}/a.pkl")
    assert not storage.exists(f"{base}/missing.pkl")
    assert storage.list_dir(base) == [f"{base}/a.pkl", f"{base}/b.pkl"]


def test_ensure_parent_is_a_no_op_on_the_api_route(monkeypatch):
    """files.upload creates the path itself; there is no directory to make."""
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    storage.set_files_client(_FakeClient())

    storage.ensure_parent(VOL)  # must not raise


# --- the routing probe --------------------------------------------------


def test_the_probe_reads_the_runtime_not_a_guess(monkeypatch):
    """`/Volumes` EXISTS in the DCS container — only listing it fails. A
    probe based on existence would call the mount healthy and send every
    write into a path that goes nowhere."""
    def _boom(path):
        raise PermissionError(1, "Operation not permitted", path)

    monkeypatch.setattr(storage.os, "listdir", _boom)
    storage.reset_route_cache()

    assert storage.posix_volumes_available() is False


def test_a_working_mount_is_detected(monkeypatch):
    monkeypatch.setattr(storage.os, "listdir", lambda path: ["forecastiq"])
    storage.reset_route_cache()

    assert storage.posix_volumes_available() is True


def test_the_probe_runs_once_per_process(monkeypatch):
    calls = []
    monkeypatch.setattr(storage.os, "listdir", lambda p: calls.append(p) or ["x"])
    storage.reset_route_cache()

    for _ in range(5):
        storage.posix_volumes_available()

    assert len(calls) == 1


# --- no silent fallback -------------------------------------------------


def test_unreachable_volume_storage_fails_it_does_not_fall_back(monkeypatch):
    """The rule this whole module exists for: a run must never report
    success with its outputs on disposable driver storage."""
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    storage.reset_client_cache()

    def _no_runtime():
        raise ImportError("no databricks runtime here")

    monkeypatch.setattr(storage, "_files_client", _no_runtime)

    with pytest.raises(Exception) as caught:
        storage.write_bytes(VOL, b"data")

    assert not isinstance(caught.value, FileNotFoundError)


def test_an_authentication_failure_raises_storage_error_naming_the_cause():
    """Outside a Databricks runtime there is no notebook context, which is
    the same shape of failure the container hit with default auth. It must
    surface as StorageError explaining what could not be reached — not as
    an opaque AttributeError from deep inside the SDK."""
    storage.reset_client_cache()

    with pytest.raises(storage.StorageError) as caught:
        storage._files_client()

    message = str(caught.value)
    assert "Unity Catalog" in message
    assert "Files API" in message


def test_a_missing_file_is_a_missing_file_not_a_storage_outage(monkeypatch):
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    storage.set_files_client(_FakeClient())

    with pytest.raises(FileNotFoundError):
        storage.read_bytes(VOL)


# --- the security guardrail --------------------------------------------


def test_no_direct_storage_access_is_possible_through_this_module():
    """UC is the authorization boundary. A storage key, SAS token or raw
    abfss:// URL would reach the bytes while stepping around the volume
    grants that protect them."""
    import ast
    import inspect

    # Parsed, not grepped: the module docstring names these mechanisms
    # precisely to say they are forbidden. Only executable code counts.
    tree = ast.parse(inspect.getsource(storage))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body[0].value.value = ""
    code = ast.unparse(tree).lower()
    for forbidden in ("abfss://", "accountkey", "sas_token", "sharedkey", "account_key", "blob.core.windows.net"):
        assert forbidden not in code, f"{forbidden} appears in executable code"


# --- "missing" and "broken" must never be confused ---------------------


class _DeniedFiles:
    """Unity Catalog refusing the caller, as observed from the DCS runtime:
    PermissionDenied: User does not have READ VOLUME on Volume '…'."""

    def get_metadata(self, path):
        raise _PermissionDenied(f"User does not have READ VOLUME on Volume '{path}'")

    def download(self, path):
        raise _PermissionDenied(f"User does not have READ VOLUME on Volume '{path}'")

    def upload(self, path, contents, overwrite=False):
        raise _PermissionDenied(f"User does not have READ VOLUME on Volume '{path}'")


class _PermissionDenied(Exception):
    pass


_PermissionDenied.__name__ = "PermissionDenied"


class _DeniedClient:
    def __init__(self):
        self.files = _DeniedFiles()


def test_an_authorization_denial_is_not_reported_as_a_missing_file(monkeypatch):
    """The engine asks exists() to decide whether to say "Dataset could not
    be found". A UC denial answered as False would tell a user their data is
    gone when the truth is that they are not allowed to read it."""
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    storage.set_files_client(_DeniedClient())

    with pytest.raises(storage.StorageError) as caught:
        storage.exists(VOL)

    assert "READ VOLUME" in str(caught.value)


def test_a_denial_on_read_and_write_also_surfaces(monkeypatch):
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    storage.set_files_client(_DeniedClient())

    with pytest.raises(storage.StorageError):
        storage.read_bytes(VOL)
    with pytest.raises(storage.StorageError):
        storage.write_bytes(VOL, b"x")


def test_a_genuinely_absent_file_still_answers_false(monkeypatch):
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    storage.set_files_client(_FakeClient())

    assert storage.exists(VOL) is False


# --- transient failure retry --------------------------------------------


class _FlakyFiles(_FakeFiles):
    """Fails a download or upload a fixed number of times before behaving
    like the real thing — standing in for a connection that drops mid
    transfer and succeeds on a fresh attempt."""

    def __init__(self, fail_times: int, error: Exception):
        super().__init__()
        self._fail_times = fail_times
        self._error = error
        self.download_attempts = 0
        self.upload_attempts = 0

    def download(self, path):
        self.download_attempts += 1
        if self.download_attempts <= self._fail_times:
            raise self._error
        return super().download(path)

    def upload(self, path, contents, overwrite=False):
        self.upload_attempts += 1
        if self.upload_attempts <= self._fail_times:
            raise self._error
        return super().upload(path, contents, overwrite=overwrite)


class _FlakyClient:
    def __init__(self, files):
        self.files = files


class _ConnectionBroken(Exception):
    """Stands in for urllib3's ChunkedEncodingError/IncompleteRead — a
    plain Exception the SDK does not classify as NotFound."""


def test_a_download_that_drops_mid_transfer_is_retried_and_succeeds(monkeypatch):
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    files = _FlakyFiles(fail_times=2, error=_ConnectionBroken("connection broken"))
    files.store[VOL] = b"payload"
    storage.set_files_client(_FlakyClient(files))

    assert storage.read_bytes(VOL) == b"payload"
    assert files.download_attempts == 3


def test_an_upload_that_drops_mid_transfer_is_retried_and_succeeds(monkeypatch):
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    files = _FlakyFiles(fail_times=1, error=_ConnectionBroken("connection broken"))
    storage.set_files_client(_FlakyClient(files))

    storage.write_bytes(VOL, b"payload")

    assert files.upload_attempts == 2
    assert files.store[VOL] == b"payload"


def test_retries_are_bounded_then_the_real_error_surfaces(monkeypatch):
    """A connection that never recovers must still fail the run — retrying
    is for a blip, not a cover for storage that is genuinely unreachable."""
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    files = _FlakyFiles(fail_times=99, error=_ConnectionBroken("connection broken"))
    files.store[VOL] = b"payload"
    storage.set_files_client(_FlakyClient(files))

    with pytest.raises(storage.StorageError):
        storage.read_bytes(VOL)

    assert files.download_attempts == storage._TRANSIENT_RETRY_ATTEMPTS


def test_a_missing_file_is_never_retried(monkeypatch):
    """NotFound means the file genuinely is not there — retrying it only
    delays reporting a real failure as if it might resolve itself."""
    monkeypatch.setenv(storage.POSIX_VOLUMES_ENV_VAR, "0")
    files = _FakeFiles()
    storage.set_files_client(_FlakyClient(files))

    with pytest.raises(FileNotFoundError):
        storage.read_bytes(VOL)

    assert files.downloads == [VOL]
