from __future__ import annotations

import inspect
from datetime import date

import polars as pl
import pytest
from scripts import run_legacy

from alpharank.production import legacy_pipeline
from alpharank.strategy.legacy_valuation import NO_SEC_FUNDAMENTALS_POLICY_ID
from alpharank.utils.frame_backend import normalize_polars_date_columns


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


@pytest.mark.parametrize(
    "dates",
    [
        [date(2026, 9, 30), date(2026, 10, 30)],
        ["2026-09-30", "2026-10-30"],
    ],
)
def test_input_date_boundary_accepts_native_and_iso_text(
    dates: list[date] | list[str],
) -> None:
    normalized = normalize_polars_date_columns(
        pl.DataFrame({"date": dates, "value": [1.0, 2.0]}),
        ("date",),
    )

    assert normalized.schema["date"] == pl.Date
    assert normalized["date"].to_list() == [date(2026, 9, 30), date(2026, 10, 30)]
    assert (
        inspect.signature(legacy_pipeline.run_pipeline)
        .parameters["fundamental_eligibility_policy_id"]
        .default
        == NO_SEC_FUNDAMENTALS_POLICY_ID
    )
