from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from alpharank.data.composed_snapshot import (
    PRICE_FILES,
    SEC_FILES,
    build_composed_model_snapshot,
    validate_composed_model_snapshot,
)


def _write_price_package(root: Path) -> None:
    root.mkdir(parents=True)
    price = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "date": ["2026-08-14"],
            "adjusted_close": [10.0],
        }
    )
    for name in PRICE_FILES:
        if name.endswith(".csv"):
            (root / name).write_text("Date,Ticker,Name\n2026-08-01,AAA,AAA\n", encoding="utf-8")
        else:
            price.write_parquet(root / name)
    manifest = {
        "run_id": "price-run",
        "source_refresh_contract": {
            "snapshot_scope": "full_ingestion",
            "policy": {"require_eodhd_price_seed": True},
            "eodhd_price_seed": {"sha256": "seed-hash"},
            "price_revision_guard": {"passed": True},
        },
        "data_freshness": {"prices": {"max_market_date": "2026-08-14"}},
    }
    (root / "lineage").mkdir()
    (root / "lineage" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _upgrade_price_package_to_persistent_v2(root: Path) -> None:
    registry_path = root / "lineage" / "persistent_price_history_registry.parquet"
    pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "row_count": [1],
            "persistence_policy_id": ["published_price_history_v1"],
        }
    ).write_parquet(registry_path)
    manifest_path = root / "lineage" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract_version"] = 2
    manifest["source_refresh_contract"]["persistent_price_history"] = {
        "policy_id": "published_price_history_v1",
        "routine_deletion_allowed": False,
    }
    manifest["validation"] = {
        "all_previous_validated_inactive_history_preserved": True,
        "open_source_only_inactive_history_persisted": True,
    }
    manifest["artifacts"] = {
        "persistent_price_history_registry": {
            "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest()
        }
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_sec_package(root: Path, *, source: str = "sec_companyfacts") -> None:
    root.mkdir(parents=True)
    frame = pl.DataFrame({"ticker": ["AAA.US"], "date": ["2026-06-30"]})
    for name in SEC_FILES:
        frame.write_parquet(root / name)
    lineage = root / "lineage"
    lineage.mkdir()
    pl.DataFrame({"ticker": ["AAA.US"], "selected_source": [source]}).write_parquet(
        lineage / "financials_sec_lineage.parquet"
    )
    pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "calendar_source": ["sec_submissions"],
            "actual_source": ["sec_companyfacts"],
        }
    ).write_parquet(lineage / "earnings_sec_lineage.parquet")
    (lineage / "manifest.json").write_text(
        json.dumps({"run_id": "sec-run", "scope": "sec_only_fundamentals"}),
        encoding="utf-8",
    )


def test_build_composed_snapshot_is_hash_verified_and_immutable(tmp_path: Path) -> None:
    price = tmp_path / "price"
    sec = tmp_path / "sec"
    _write_price_package(price)
    _write_sec_package(sec)

    result = build_composed_model_snapshot(
        price_package_dir=price,
        sec_package_dir=sec,
        history_root=tmp_path / "history",
        latest_manifest_path=tmp_path / "latest.json",
        expected_through="2026-08-16",
    )

    assert result.snapshot_dir.exists()
    assert result.manifest["validation"]["passed"] is True
    assert validate_composed_model_snapshot(result.snapshot_dir)["passed"] is True
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["composition_id"] == result.composition_id


def test_composed_snapshot_rejects_non_sec_fundamental_source(tmp_path: Path) -> None:
    price = tmp_path / "price"
    sec = tmp_path / "sec"
    _write_price_package(price)
    _write_sec_package(sec, source="yfinance")

    with pytest.raises(RuntimeError, match="forbidden sources"):
        build_composed_model_snapshot(
            price_package_dir=price,
            sec_package_dir=sec,
            history_root=tmp_path / "history",
        )


def test_composed_snapshot_rejects_stale_prices(tmp_path: Path) -> None:
    price = tmp_path / "price"
    sec = tmp_path / "sec"
    _write_price_package(price)
    _write_sec_package(sec)
    manifest_path = price / "lineage" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data_freshness"]["prices"]["max_market_date"] = "2026-07-01"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Invalid market freshness date"):
        build_composed_model_snapshot(
            price_package_dir=price,
            sec_package_dir=sec,
            history_root=tmp_path / "history",
            expected_through="2026-08-16",
        )


def test_composed_snapshot_rejects_prelisting_data_on_reused_symbol(
    tmp_path: Path,
) -> None:
    price = tmp_path / "price"
    sec = tmp_path / "sec"
    _write_price_package(price)
    _write_sec_package(sec)
    pl.DataFrame(
        {
            "ticker": ["SNDK.US"],
            "date": ["2016-04-03"],
            "adjusted_close": [75.0],
        }
    ).write_parquet(price / "US_Finalprice.parquet")
    (price / "SP500_Constituents.csv").write_text(
        "Date,Ticker,Name\n2016-04-01,SNDK,SanDisk\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="security identity interval"):
        build_composed_model_snapshot(
            price_package_dir=price,
            sec_package_dir=sec,
            history_root=tmp_path / "history",
        )


def test_composed_snapshot_rejects_v2_price_without_persistent_history(tmp_path: Path) -> None:
    price = tmp_path / "price"
    sec = tmp_path / "sec"
    _write_price_package(price)
    _write_sec_package(sec)
    manifest_path = price / "lineage" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract_version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="persistent-history contract"):
        build_composed_model_snapshot(
            price_package_dir=price,
            sec_package_dir=sec,
            history_root=tmp_path / "history",
        )


def test_price_registry_promotion_preserves_payload(tmp_path: Path) -> None:
    price = tmp_path / "price"
    sec = tmp_path / "sec"
    _write_price_package(price)
    _upgrade_price_package_to_persistent_v2(price)
    _write_sec_package(sec)

    source_path = price / "US_Finalprice.parquet"
    source_bytes = source_path.read_bytes()
    source_frame = pl.read_parquet(source_path)
    result = build_composed_model_snapshot(
        price_package_dir=price,
        sec_package_dir=sec,
        history_root=tmp_path / "history",
        latest_manifest_path=tmp_path / "latest.json",
    )

    promoted_path = result.snapshot_dir / "US_Finalprice.parquet"
    promoted_frame = pl.read_parquet(promoted_path)
    identity = result.manifest["price_payload_identity"]
    assert promoted_path.read_bytes() == source_bytes
    assert promoted_frame.equals(source_frame, null_equal=True)
    assert identity["row_count"] == source_frame.height
    assert identity["unique_key_count"] == source_frame.height
    assert identity["duplicate_key_count"] == 0
    assert result.manifest["validation"]["persistent_price_registry_copied"] is True
    assert (
        result.snapshot_dir / "lineage" / "prices" / "persistent_price_history_registry.parquet"
    ).is_file()
    assert validate_composed_model_snapshot(result.snapshot_dir)["passed"] is True
