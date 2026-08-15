from pathlib import Path

from fastapi import UploadFile

from app.config.settings import get_settings
from app.schemas.upload import UploadResponse
from app.utils.exceptions import FileResolutionError
from app.utils.ids import generate_file_id


class UploadService:
    """Saves uploads and resolves a file_id back to a path.

    The only place that knows the {file_id}_{filename} naming convention.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def save(self, file: UploadFile) -> UploadResponse:
        """Save an upload and return the file_id later requests refer to it by."""
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
        """The path and original filename for a file_id.

        Raises FileResolutionError if nothing staged matches it.
        """
        matches = list(self._settings.upload_path.glob(f"{file_id}_*"))
        if not matches:
            raise FileResolutionError(f"No uploaded file found for file_id '{file_id}'.")

        path = matches[0]
        original_filename = path.name[len(file_id) + 1 :]
        return path, original_filename
