#!/usr/bin/env python3
"""Catalogue historical data roots by reference and SHA-256 before migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.data.warehouse.historical_migration import (
    build_historical_catalog_summary,
    build_historical_root_catalog,
    validate_historical_root_catalog,
    write_historical_catalog_summary,
    write_historical_root_catalog,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "docs" / "architecture" / "historical_data_migration_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--generated-at", default="2026-08-20")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.validate_only and args.catalog is None:
        parser.error("--validate-only requires --catalog")
    if args.catalog is None:
        catalog_root = (
            root
            / "data"
            / "warehouse"
            / "manifests"
            / "historical_migrations"
        )
        temporary_catalog = build_historical_root_catalog(
            _historical_roots(root),
            generated_at=args.generated_at,
        )
        catalog_path = (
            catalog_root
            / str(temporary_catalog["catalog_id"])
            / "manifest.json"
        )
        if not args.validate_only:
            write_historical_root_catalog(catalog_path, temporary_catalog)
    else:
        catalog_path = args.catalog.resolve()
    validation = validate_historical_root_catalog(catalog_path)
    if not args.validate_only:
        write_historical_catalog_summary(
            args.summary.resolve(),
            build_historical_catalog_summary(catalog_path),
        )
    print(json.dumps(validation, indent=2, sort_keys=True))


def _historical_roots(root: Path) -> dict[str, Path]:
    data = root / "data"
    return {
        "legacy_balance": data / "US_Balance_sheet.parquet",
        "legacy_benchmark": data / "SP500Price.parquet",
        "legacy_cash_flow": data / "US_Cash_flow.parquet",
        "legacy_constituents": data / "SP500_Constituents.csv",
        "legacy_earnings": data / "US_Earnings.parquet",
        "legacy_final_price": data / "US_Finalprice.parquet",
        "legacy_general": data / "US_General.parquet",
        "legacy_income": data / "US_Income_statement.parquet",
        "legacy_latest_pointer": data / "latest_snapshot.json",
        "legacy_shares": data / "US_share.parquet",
        "local_snapshots": data / "_snapshots",
        "eodhd_archive": data / "eodhd",
        "open_source_archive": data / "open_source" / "archive",
        "open_source_history": data / "open_source" / "history",
        "open_source_official": data / "open_source" / "official",
        "open_source_output": data / "open_source" / "output",
        "sec_legacy": data / "sec",
        "legacy_data_outputs": data / "outputs",
    }


if __name__ == "__main__":
    main()
