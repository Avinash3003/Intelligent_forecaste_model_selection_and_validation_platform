import uuid
from datetime import datetime, timezone


def generate_file_id() -> str:
    return f"file-{uuid.uuid4().hex[:12]}"


def generate_run_id() -> str:
    return f"db-run-{uuid.uuid4().hex[:8]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
