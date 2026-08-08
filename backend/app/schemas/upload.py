from pydantic import BaseModel


class UploadResponse(BaseModel):
    success: bool
    file_id: str
    filename: str
    size_bytes: int
    message: str
