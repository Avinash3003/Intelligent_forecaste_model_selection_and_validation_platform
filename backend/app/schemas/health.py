from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str
    environment: str
