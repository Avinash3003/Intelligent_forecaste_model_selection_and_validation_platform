"""Domain-specific exceptions shared across services.

Keeping these in one place lets API routers catch a single, predictable
exception type — instead of raw pandas/OS errors — and translate it into a
clean HTTP response the frontend can display directly to the user.
"""


class DatasetLoadError(Exception):
    """Raised when an uploaded file cannot be read into a usable DataFrame.

    Covers missing files, unsupported formats, corrupt/unparseable content,
    and empty datasets. The message is written to be shown to the end user
    as-is, so raise it with a complete sentence.
    """


class FileResolutionError(Exception):
    """Raised when a `file_id` sent by the frontend doesn't match any file
    previously staged by UploadService — e.g. it was never uploaded, or
    local storage was cleared since the upload happened.
    """


class UploadTooLargeError(Exception):
    """Raised when an upload exceeds MAX_UPLOAD_SIZE_MB.

    Detected while streaming, so the request is refused without ever holding
    the whole file. The message states the limit and is shown to the user
    as-is.
    """
