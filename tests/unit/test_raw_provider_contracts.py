from __future__ import annotations

from pathlib import Path

import pytest

from alpharank.data.raw_contracts import (
    load_raw_provider_contracts,
    provider_contract,
)
from alpharank.data.warehouse import WarehousePaths


def test_every_raw_provider_targets_its_own_warehouse_root() -> None:
    contracts = load_raw_provider_contracts()
    providers = contracts["providers"]

    assert {provider["provider_id"] for provider in providers} == {
        "eodhd",
        "sec_companyfacts",
        "sec_filings",
        "sec_submissions",
        "simfin",
        "sp500_membership",
        "stockanalysis",
        "yfinance",
    }
    assert all(
        provider["target_root"]
        == f"data/warehouse/raw/{provider['provider_id']}"
        for provider in providers
    )
    assert all(provider["datasets"] for provider in providers)
    assert contracts["provider_manifest_required_fields"] == [
        "contract",
        "provider_id",
        "dataset_id",
        "receipt_count",
        "latest_receipt_id",
        "payload_object_count",
        "generated_at",
        "receipts_sha256",
        "validation",
    ]


def test_eodhd_is_catalogued_without_authorizing_a_new_download() -> None:
    contract = provider_contract("eodhd")

    assert contract["source_kind"] == "immutable_local_archive"
    assert contract["live_download_allowed"] is False
    assert contract["migration_status"] == "catalogued_by_hash"
    assert contract["catalog_evidence"] == {
        "catalog_id": "5a1bb6261807f01c08ac3635e9d40363004e8be1428e180ef1cd186e140522a5",
        "catalog_manifest_sha256": "fd9f96abf7630aad121b3155390631576042093981e7f64719d5bda05559425b",
        "source_file_count": 49,
        "unique_object_count": 24,
    }


def test_warehouse_provider_paths_fail_closed_on_invalid_ids(tmp_path: Path) -> None:
    paths = WarehousePaths(tmp_path / "warehouse")

    assert paths.raw_provider("sec_companyfacts") == (
        tmp_path / "warehouse/raw/sec_companyfacts"
    )
    with pytest.raises(ValueError, match="Invalid RAW provider id"):
        paths.raw_provider("../outside")
