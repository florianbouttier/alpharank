from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from alpharank.data.open_source.sec_mapping import resolve_sec_company_mapping
from alpharank.data.security_identity import (
    apply_security_identity_policy,
    apply_security_identity_reference_policy,
    assert_security_identity_compliance,
    load_security_identity_registry,
)


def _registry() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_ticker": ["SNDK", "SNDK"],
            "canonical_ticker": ["SNDK_OLD", "SNDK"],
            "security_id": ["sec-cik-0001000180", "sec-cik-0002023554"],
            "issuer_cik": ["0001000180", "0002023554"],
            "valid_from": ["2005-01-01", "2025-02-24"],
            "valid_to": ["2016-05-12", None],
            "identity_status": ["historical", "current"],
            "evidence": ["fixture-old", "fixture-new"],
        }
    )


def test_symbol_reuse_maps_rows_to_distinct_security_identities() -> None:
    source = pl.DataFrame(
        {
            "ticker": ["SNDK.US", "SNDK.US", "SNDK.US"],
            "date": ["2016-05-12", "2025-02-20", "2025-02-24"],
            "adjusted_close": [75.0, 48.0, 50.0],
        }
    )

    result = apply_security_identity_policy(
        source,
        ticker_column="ticker",
        date_column="date",
        registry=_registry(),
    )

    assert result.frame.select("ticker", "date").to_dicts() == [
        {"ticker": "SNDK_OLD.US", "date": "2016-05-12"},
        {"ticker": "SNDK.US", "date": "2025-02-24"},
    ]
    assert result.rejected.select("ticker", "date").to_dicts() == [
        {"ticker": "SNDK.US", "date": "2025-02-20"}
    ]
    assert result.report["security_identity_count"] == 2
    assert result.report["rejected_rows"] == 1


def test_identity_compliance_rejects_prelisting_new_sndk_row() -> None:
    contaminated = pl.DataFrame(
        {
            "ticker": ["SNDK.US"],
            "date": ["2016-04-03"],
        }
    )

    with pytest.raises(RuntimeError, match="security identity interval"):
        assert_security_identity_compliance(
            contaminated,
            ticker_column="ticker",
            date_column="date",
            registry=_registry(),
        )


def test_reference_rows_use_distinct_canonical_tickers_for_each_cik() -> None:
    source = pl.DataFrame(
        {
            "ticker": ["SNDK.US", "SNDK.US", "SNDK.US"],
            "cik": ["0001000180", "0002023554", "0000000001"],
            "name": ["Old SanDisk", "New Sandisk", "Wrong issuer"],
        }
    )

    result = apply_security_identity_reference_policy(
        source,
        ticker_column="ticker",
        registry=_registry(),
    )

    assert result.frame.select("ticker", "cik").to_dicts() == [
        {"ticker": "SNDK_OLD.US", "cik": "0001000180"},
        {"ticker": "SNDK.US", "cik": "0002023554"},
    ]
    assert result.rejected.get_column("name").to_list() == ["Wrong issuer"]


def test_real_sndk_registry_separates_old_and_new_ciks() -> None:
    registry = load_security_identity_registry()
    sndk = registry.filter(pl.col("source_ticker") == "SNDK").sort("valid_from")

    assert sndk.select("canonical_ticker", "issuer_cik").to_dicts() == [
        {"canonical_ticker": "SNDK_OLD", "issuer_cik": "0001000180"},
        {"canonical_ticker": "SNDK", "issuer_cik": "0002023554"},
    ]
    assert sndk.row(0, named=True)["valid_to"] < sndk.row(1, named=True)["valid_from"]
    assert Path(sndk.row(0, named=True)["registry_path"]).is_file()


def test_active_sec_mapping_cannot_fall_back_to_old_sndk_cik() -> None:
    live = pl.DataFrame(
        {
            "ticker": ["SNDK"],
            "name": ["Sandisk Corporation"],
            "exchange": ["Nasdaq"],
            "cik": [2023554],
        }
    )

    mapping = resolve_sec_company_mapping(
        requested_tickers=["SNDK"],
        sec_mapping_all=live,
    )

    assert mapping.select("ticker", "cik", "mapping_source").to_dicts() == [
        {
            "ticker": "SNDK",
            "cik": "0002023554",
            "mapping_source": "sec_live_mapping",
        }
    ]
