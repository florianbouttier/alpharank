from __future__ import annotations

import inspect

from scripts import run_legacy

from alpharank.production import legacy_pipeline
from alpharank.strategy.legacy_valuation import NO_SEC_FUNDAMENTALS_POLICY_ID


def test_run_legacy_is_a_thin_public_command() -> None:
    assert run_legacy.run_pipeline is legacy_pipeline.run_pipeline
    assert len(inspect.getsource(run_legacy).splitlines()) < 350


def test_legacy_pipeline_runtime_provenance_hashes_its_owner() -> None:
    source = inspect.getsource(legacy_pipeline.run_pipeline)
    assert '"src/alpharank/production/legacy_pipeline.py"' in source


def test_no_sec_fundamentals_is_the_canonical_legacy_default() -> None:
    assert (
        inspect.signature(run_legacy.main)
        .parameters["fundamental_eligibility_policy_id"]
        .default
        == NO_SEC_FUNDAMENTALS_POLICY_ID
    )
    assert (
        inspect.signature(legacy_pipeline.run_pipeline)
        .parameters["fundamental_eligibility_policy_id"]
        .default
        == NO_SEC_FUNDAMENTALS_POLICY_ID
    )
