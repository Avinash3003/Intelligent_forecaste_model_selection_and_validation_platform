"""The `--config` payload must be able to set pipeline-level blocks.

`PipelineConfig.from_dict` has always understood `curated_storage`, but
`main()` never built a PipelineConfig from the payload, so the engine always
used the default *relative* curated root. On a Databricks driver that
resolves against a working directory the job destroys on exit, so the
curated dataset did not outlive the run.
"""

from forecast_engine.config.pipeline_config import CuratedStorageConfig, PipelineConfig


def test_curated_root_comes_from_the_config_payload():
    config = PipelineConfig.from_dict(
        {"curated_storage": {"root_dir": "/Volumes/cat/sch/curated_files/runs/dbx-run-1"}}
    )
    assert config.curated_storage.root_dir == "/Volumes/cat/sch/curated_files/runs/dbx-run-1"
    assert config.curated_storage.enabled is True


def test_default_curated_root_is_unchanged_for_local_runs():
    # Local execution keeps the relative default, which is correct there:
    # the filesystem outlives the process.
    assert CuratedStorageConfig().root_dir == "curated"
    assert PipelineConfig.default().curated_storage.root_dir == "curated"


def test_an_empty_payload_leaves_every_default_intact():
    assert PipelineConfig.from_dict({}).curated_storage.root_dir == "curated"


def test_unrelated_payload_keys_do_not_disturb_curated_storage():
    # The same payload carries column mapping and run options; only the
    # blocks PipelineConfig knows about should be consumed.
    config = PipelineConfig.from_dict(
        {"date_column": "date", "target_column": "sales", "run_id": "x", "models": ["prophet"]}
    )
    assert config.curated_storage.root_dir == "curated"
