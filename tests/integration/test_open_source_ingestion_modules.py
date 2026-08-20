from __future__ import annotations

import inspect

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
