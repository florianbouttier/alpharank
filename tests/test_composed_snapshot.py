from __future__ import annotations

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
            (root / name).write_text(
                "Date,Ticker,Name\n2026-08-01,AAA,AAA\n", encoding="utf-8"
            )
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
    (root / "lineage" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _write_sec_package(root: Path, *, source: str = "sec_companyfacts") -> None:
    root.mkdir(parents=True)
    frame = pl.DataFrame({"ticker": ["AAA.US"], "date": ["2026-06-30"]})
    for name in SEC_FILES:
        frame.write_parquet(root / name)
    lineage = root / "lineage"
    lineage.mkdir()
    pl.DataFrame(
        {"ticker": ["AAA.US"], "selected_source": [source]}
    ).write_parquet(lineage / "financials_sec_lineage.parquet")
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
