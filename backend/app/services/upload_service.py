from pathlib import Path

from fastapi import UploadFile

from app.config.settings import get_settings
from app.schemas.upload import UploadResponse
from app.utils.exceptions import FileResolutionError, UploadTooLargeError
from app.utils.ids import generate_file_id

_CHUNK_BYTES = 1024 * 1024


class UploadService:
    """Saves uploads and resolves a file_id back to a path.

    The only place that knows the {file_id}_{filename} naming convention.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def save(self, file: UploadFile) -> UploadResponse:
        """Save an upload and return the file_id later requests refer to it by.

        Streamed in chunks rather than read whole: a dataset is bounded by
        MAX_UPLOAD_SIZE_MB, not by the memory of the process receiving it, and
        reading it into a bytes object first held the entire file — twice,
        counting the write. Memory here is one chunk regardless of file size.

        Raises UploadTooLargeError once the limit is passed, before the rest
        of the body is read, leaving nothing partial behind.
        """
        file_id = generate_file_id()
        destination = self._settings.upload_path / f"{file_id}_{file.filename}"
        limit = self._settings.max_upload_size_mb * 1024 * 1024

        written = 0
        try:
            with destination.open("wb") as sink:
                while chunk := file.file.read(_CHUNK_BYTES):
                    written += len(chunk)
                    if written > limit:
                        raise UploadTooLargeError(
                            f"This file is larger than the {self._settings.max_upload_size_mb} MB "
                            "upload limit. Split it, or ask an administrator to raise the limit."
                        )
                    sink.write(chunk)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise

        return UploadResponse(
            success=True,
            file_id=file_id,
            filename=file.filename or "unknown",
            size_bytes=written,
            message="File uploaded and staged successfully.",
        )

    def resolve(self, file_id: str) -> tuple[Path, str]:
        """The path and original filename for a file_id.

        Raises FileResolutionError if nothing staged matches it.
        """
        matches = list(self._settings.upload_path.glob(f"{file_id}_*"))
        if not matches:
            raise FileResolutionError(f"No uploaded file found for file_id '{file_id}'.")

        path = matches[0]
        original_filename = path.name[len(file_id) + 1 :]
        return path, original_filename
