"""Uploads stream to disk and are bounded by the configured limit.

A 16 MB dataset intermittently failed with "the request took too long":
the client aborted at a fixed 30s regardless of file size, which is the
file size over the user's upstream bandwidth. That half is fixed in
apiClient. This half is the server: the body was read whole into memory
(`file.file.read()`), so a large dataset was held twice, and
MAX_UPLOAD_SIZE_MB was declared but never enforced.
"""

from __future__ import annotations

import io

import pytest
from fastapi import UploadFile

from app.config.settings import Settings
from app.services.upload_service import UploadService
from app.utils.exceptions import UploadTooLargeError


def _service(tmp_path, limit_mb=200):
    service = UploadService()
    service._settings = Settings(upload_dir=str(tmp_path), max_upload_size_mb=limit_mb)
    return service


def _upload(payload: bytes, name="sales.csv") -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(payload))


def test_a_file_is_written_whole_and_its_size_reported(tmp_path):
    payload = b"date,sales\n" + b"2024-01-01,10\n" * 50_000
    result = _service(tmp_path).save(_upload(payload))

    assert result.success is True
    assert result.size_bytes == len(payload)
    staged = next(tmp_path.glob(f"{result.file_id}_*"))
    assert staged.read_bytes() == payload


def test_the_body_is_never_read_whole_into_memory(tmp_path):
    """The defect this replaced: one read() of the entire upload."""
    reads: list[int] = []

    class _Counting(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    payload = b"x" * (5 * 1024 * 1024)
    _service(tmp_path).save(UploadFile(filename="big.csv", file=_Counting(payload)))

    assert reads, "the service never read the body"
    assert all(size > 0 for size in reads), "a read(-1) pulls the whole file into memory"
    assert max(reads) <= 1024 * 1024


def test_an_oversized_upload_is_refused(tmp_path):
    service = _service(tmp_path, limit_mb=1)

    with pytest.raises(UploadTooLargeError, match="1 MB"):
        service.save(_upload(b"y" * (2 * 1024 * 1024)))


def test_a_refused_upload_leaves_nothing_behind(tmp_path):
    service = _service(tmp_path, limit_mb=1)

    with pytest.raises(UploadTooLargeError):
        service.save(_upload(b"y" * (2 * 1024 * 1024)))

    assert list(tmp_path.iterdir()) == []


def test_a_file_at_the_limit_is_accepted(tmp_path):
    service = _service(tmp_path, limit_mb=1)

    result = service.save(_upload(b"z" * (1024 * 1024)))

    assert result.size_bytes == 1024 * 1024


def test_the_staged_file_resolves_back_to_its_original_name(tmp_path):
    service = _service(tmp_path)
    result = service.save(_upload(b"date,sales\n2024-01-01,1\n", name="my dataset.csv"))

    path, original = service.resolve(result.file_id)

    assert original == "my dataset.csv"
    assert path.exists()
