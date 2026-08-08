from pathlib import Path

from fastapi import UploadFile

from app.config.settings import get_settings
from app.schemas.upload import UploadResponse
from app.utils.exceptions import FileResolutionError
from app.utils.ids import generate_file_id


class UploadService:
    """Stores uploaded datasets to a local temp directory and resolves a
    `file_id` back to a path on disk. This is the ONLY place that knows the
    on-disk naming convention (`{file_id}_{original_filename}`) — swap it
    for Azure Data Lake / Blob Storage in a later phase without touching
    any other service."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def save(self, file: UploadFile) -> UploadResponse:
        """Persist an uploaded file locally and return a reference to it.

        Args:
            file: The multipart file submitted to POST /upload.

        Returns:
            An UploadResponse carrying the generated `file_id` the frontend
            must send back in later requests (e.g. POST /metadata/validate).
        """
        file_id = generate_file_id()
        destination = self._settings.upload_path / f"{file_id}_{file.filename}"

        contents = file.file.read()
        destination.write_bytes(contents)

        return UploadResponse(
            success=True,
            file_id=file_id,
            filename=file.filename or "unknown",
            size_bytes=len(contents),
            message="File uploaded and staged successfully.",
        )

    def resolve(self, file_id: str) -> tuple[Path, str]:
        """Resolve a previously issued `file_id` back to its file path and
        original filename.

        Called by the /metadata/validate route before the dataset can be
        loaded and interpreted.

        Raises:
            FileResolutionError: if no staged file matches `file_id` (e.g.
                an invalid id, or local storage was cleared since upload).
        """
        matches = list(self._settings.upload_path.glob(f"{file_id}_*"))
        if not matches:
            raise FileResolutionError(f"No uploaded file found for file_id '{file_id}'.")

        path = matches[0]
        original_filename = path.name[len(file_id) + 1 :]
        return path, original_filename
