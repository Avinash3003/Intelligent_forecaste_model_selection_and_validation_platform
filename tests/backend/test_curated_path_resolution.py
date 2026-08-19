"""A run's curated dataset stays readable after the app moves.

App Service runs the backend from a per-build /tmp directory, so an absolute
path a run recorded under one deployment does not exist under the next. The
run itself is fine — its curated file is in a UC volume — but resolution used
to try only the recorded path, so the Results page rendered a run with no
data after any restart or redeploy.

Resolution now tries the recorded path first, then (cloud runs only) the
location that file occupies in the configured curated volume today. Local
execution is deliberately untouched: its recorded path IS the authoritative
one and there is no volume to fall back to.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.services.dataset_preview_service import DatasetPreviewService

CURATED_ROOT = "/Volumes/forecastiq/forecasting/curated_files"
RUN = "fe-run-abc123"
FILENAME = "curated_file-deadbeef_sales.csv"
DURABLE = f"{CURATED_ROOT}/runs/{RUN}/{FILENAME}"
STALE_TMP = f"/tmp/8defc9a91cf611a/curated/{RUN}/{FILENAME}"


def _service(mode: str) -> DatasetPreviewService:
    return DatasetPreviewService(
        executor=object(),
        settings=Settings(
            execution_mode=mode,
            databricks_curated_volumes_root=CURATED_ROOT,
        ),
    )


def test_databricks_rebuilds_the_volume_path_from_a_stale_tmp_path():
    """The bug this fixes: a path recorded under a previous build."""
    candidates = _service("databricks")._candidate_uris(STALE_TMP, RUN)

    assert candidates == [STALE_TMP, DURABLE]


def test_the_recorded_path_is_always_tried_first():
    """A run whose recorded path still resolves must not be redirected."""
    assert _service("databricks")._candidate_uris(DURABLE, RUN)[0] == DURABLE


def test_an_already_durable_path_is_not_duplicated():
    assert _service("databricks")._candidate_uris(DURABLE, RUN) == [DURABLE]


def test_local_execution_keeps_exactly_one_candidate():
    """Local's recorded path is authoritative — no volume exists to fall back
    to, so its behaviour is unchanged."""
    local_path = "/home/sigmoid/Documents/tech_demo/curated/fe-run-abc123/x.csv"
    assert _service("local")._candidate_uris(local_path, RUN) == [local_path]


def test_a_run_id_is_required_to_rebuild():
    """Without the run id the layout cannot be reconstructed, so nothing is
    guessed — the recorded path stands alone."""
    assert _service("databricks")._candidate_uris(STALE_TMP, None) == [STALE_TMP]


def test_an_unconfigured_volume_root_adds_no_candidate():
    service = DatasetPreviewService(
        executor=object(),
        settings=Settings(execution_mode="databricks", databricks_curated_volumes_root=""),
    )
    assert service._candidate_uris(STALE_TMP, RUN) == [STALE_TMP]


def test_the_root_comes_from_configuration_not_a_constant():
    """Retargeting a workspace is a settings change, never a code change."""
    other = "/Volumes/other_cat/other_schema/curated_files"
    service = DatasetPreviewService(
        executor=object(),
        settings=Settings(execution_mode="databricks", databricks_curated_volumes_root=other),
    )
    assert service._candidate_uris(STALE_TMP, RUN)[1] == f"{other}/runs/{RUN}/{FILENAME}"


def test_resolution_is_deterministic_across_processes():
    """A fresh service instance — i.e. a new process after a restart —
    resolves to exactly the same durable location."""
    first = _service("databricks")._candidate_uris(STALE_TMP, RUN)
    second = _service("databricks")._candidate_uris(STALE_TMP, RUN)
    assert first == second == [STALE_TMP, DURABLE]


def test_a_missing_file_everywhere_reads_as_none_not_an_error():
    """Neither candidate resolves and there is no workspace client: the
    preview reports itself unavailable rather than raising."""
    assert _service("databricks")._read(STALE_TMP, RUN) is None


def test_local_missing_file_also_reads_as_none():
    assert _service("local")._read("/nonexistent/path/x.csv", RUN) is None
