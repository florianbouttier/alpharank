from __future__ import annotations

import inspect

import polars as pl
import pytest

from alpharank.data.ingestion import orchestration
from alpharank.data.open_source import (
    ingestion,
    ingestion_frames,
    ingestion_prices,
    ingestion_reference,
)


def test_ingestion_facade_preserves_stage_function_identities() -> None:
    assert ingestion._with_price_ingestion_metadata is ingestion_frames._with_price_ingestion_metadata
    assert ingestion._consolidate_price_sources is ingestion_prices._consolidate_price_sources
    assert ingestion._fetch_sec_companyfacts_bundle is ingestion_reference._fetch_sec_companyfacts_bundle


def test_stage_modules_do_not_depend_on_ingestion_orchestration() -> None:
    for module in (ingestion_frames, ingestion_prices, ingestion_reference):
        assert "open_source.ingestion import" not in inspect.getsource(module)


def test_reference_refresh_resolves_active_universe_and_price_cutoff() -> None:
    assert orchestration._resolve_active_reference_tickers(
        ("ACTIVE", "INACTIVE"),
        ("ACTIVE", "OTHER"),
    ) == ("ACTIVE",)
    assert orchestration._resolve_active_reference_tickers(
        ("HISTORICAL",),
        (),
    ) == ("HISTORICAL",)
    assert orchestration._latest_validated_price_date(
        pl.DataFrame({"date": ["2026-08-21", "2026-08-22"]})
    ) == "2026-08-22"

    with pytest.raises(RuntimeError, match="non-empty validated price history"):
        orchestration._latest_validated_price_date(
            pl.DataFrame({"date": []}, schema={"date": pl.String})
        )
