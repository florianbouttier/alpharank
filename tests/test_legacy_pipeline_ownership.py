from __future__ import annotations

import inspect

from scripts import run_legacy

from alpharank.production import legacy_pipeline


def test_run_legacy_is_a_thin_public_command() -> None:
    assert run_legacy.run_pipeline is legacy_pipeline.run_pipeline
    assert len(inspect.getsource(run_legacy).splitlines()) < 350


def test_legacy_pipeline_runtime_provenance_hashes_its_owner() -> None:
    source = inspect.getsource(legacy_pipeline.run_pipeline)
    assert '"src/alpharank/production/legacy_pipeline.py"' in source
