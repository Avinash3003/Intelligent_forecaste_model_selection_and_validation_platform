"""Dataset preview for the Results page.

Shows the user the file they actually uploaded, alongside the decision made
from it. The blob is read from ADLS with a container-scoped read-only SAS
where one is configured, and from the local upload directory otherwise — so
the same endpoint serves both the local-execution and Azure deployments
without the frontend knowing which is in play.

Only the first slice of the file is ever fetched. A preview is for confirming
"is this the right data", not for browsing a 17MB CSV in a browser tab.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from app.config.settings import Settings, get_settings
from app.orchestration.executor import PipelineExecutor, get_pipeline_executor
from app.schemas.dataset_preview import DatasetPreview

# Enough rows to recognise the file and spot an obvious mapping mistake.
PREVIEW_ROWS = 50
# Bytes pulled from the blob — sized to the 50 rows actually rendered rather
# than to a round number. 50 rows of a typical wide CSV is well under 32KB,
# and over a cross-region link the transfer is what the user waits on.
PREVIEW_BYTES = 32_000


class DatasetPreviewService:
    """Reads the head of a run's source dataset."""

    def __init__(
        self,
        settings: Settings | None = None,
        executor: PipelineExecutor | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._executor = executor or get_pipeline_executor()
        # Built once and reused: constructing a BlobServiceClient per request
        # re-does the TLS handshake, which dominated the call (~5.5s measured
        # against ~0.3s once the client is kept).
        self._blob_service = None

    def get_preview(self, run_id: str) -> DatasetPreview:
        name = self._dataset_name(run_id)
        if not name:
            return DatasetPreview(available=False, status="This run has no recorded dataset name.")

        raw, source = self._read(name)
        if raw is None:
            return DatasetPreview(
                available=False,
                dataset_name=name,
                status=f"'{name}' could not be read from {source}.",
            )
        return self._parse(name, raw, source)

    # ------------------------------------------------------------------
    # Locating the file
    # ------------------------------------------------------------------

    def _dataset_name(self, run_id: str) -> str | None:
        try:
            result = self._executor.get_result(run_id)
        except Exception:  # noqa: BLE001 - an unreadable run simply has no preview
            return None
        # dataset_path is what the engine recorded for the run; only its
        # basename is needed, since the blob is addressed by name inside the
        # uploads container.
        path = (result.run_metadata or {}).get("dataset_path") or ""
        return Path(path).name if path else None

    def _read(self, name: str) -> tuple[bytes | None, str]:
        """Try ADLS first, then the local upload directory."""
        blob, source = self._read_blob(name)
        if blob is not None:
            return blob, source

        local = Path(self._settings.upload_dir) / name
        if local.is_file():
            with local.open("rb") as handle:
                return handle.read(PREVIEW_BYTES), "local upload directory"
        return None, source

    def _read_blob(self, name: str) -> tuple[bytes | None, str]:
        account = self._settings.azure_storage_account
        sas = self._settings.azure_storage_sas_token
        connection = self._settings.azure_storage_connection_string
        if not (account and sas) and not connection:
            return None, "Azure storage (not configured)"

        try:
            if self._blob_service is None:
                from azure.storage.blob import BlobServiceClient

                self._blob_service = (
                    BlobServiceClient(
                        account_url=f"https://{account}.blob.core.windows.net",
                        credential=sas,
                    )
                    if account and sas
                    else BlobServiceClient.from_connection_string(connection)
                )

            blob = self._blob_service.get_blob_client(self._settings.azure_uploads_container, name)
            # A ranged download, so file size never determines request cost.
            return blob.download_blob(offset=0, length=PREVIEW_BYTES).readall(), "Azure Blob Storage"
        except Exception:  # noqa: BLE001 - fall through to the local copy
            return None, "Azure Blob Storage"

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, name: str, raw: bytes, source: str) -> DatasetPreview:
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()

        # The final line of a ranged read is almost certainly cut mid-row;
        # dropping it avoids rendering a half-parsed record as though it were
        # real data.
        if len(lines) > 1:
            lines = lines[:-1]

        reader = csv.reader(io.StringIO("\n".join(lines)))
        rows = list(reader)
        if not rows:
            return DatasetPreview(
                available=False, dataset_name=name, status="The file appears to be empty."
            )

        header, body = rows[0], rows[1 : PREVIEW_ROWS + 1]
        return DatasetPreview(
            available=True,
            dataset_name=name,
            source=source,
            columns=header,
            rows=body,
            preview_row_count=len(body),
            truncated=len(raw) >= PREVIEW_BYTES,
        )


_service: DatasetPreviewService | None = None


def get_dataset_preview_service() -> DatasetPreviewService:
    global _service
    if _service is None:
        _service = DatasetPreviewService()
    return _service
