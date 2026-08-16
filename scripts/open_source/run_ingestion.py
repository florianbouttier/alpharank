#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from alpharank.data.open_source import run_open_source_ingestion
from alpharank.data.open_source.refresh_policy import PRODUCTION_SOURCE_REFRESH_POLICY


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified bootstrap/daily/audit ingestion for open-source market data.")
    parser.add_argument("--mode", choices=("bootstrap", "daily"), default="daily")
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker roots without .US suffix.")
    parser.add_argument("--live-dir", type=Path, default=None)
    parser.add_argument("--reference-data-dir", type=Path, default=None)
    parser.add_argument(
        "--eodhd-price-seed-path",
        type=Path,
        default=None,
        help="Immutable EODHD seed used to preserve delisted price history.",
    )
    parser.add_argument("--price-lookback-days", type=int, default=7)
    parser.add_argument("--financial-lookback-years", type=int, default=2)
    parser.add_argument("--audit-years", nargs="*", type=int, default=())
    parser.add_argument("--threshold-pct", type=float, default=0.5)
    parser.add_argument("--simfin-api-key", default=None)
    parser.add_argument("--user-agent", default="Florian Bouttier florianbouttier@example.com")
    parser.add_argument(
        "--allow-historical-revisions",
        action="store_true",
        help="Publish after explicit review even when fundamentals older than two years changed.",
    )
    parser.add_argument(
        "--allow-historical-price-revisions",
        action="store_true",
        help="Migration-only override for reviewed historical adjusted-return revisions.",
    )
    parser.add_argument(
        "--allow-historical-price-key-removals",
        action="store_true",
        help="Migration-only override for reviewed historical price row removals.",
    )
    args = parser.parse_args()

    result = run_open_source_ingestion(
        mode=args.mode,
        start_date=args.start_date,
        end_date=args.end_date,
        tickers=args.tickers,
        live_dir=args.live_dir.resolve() if args.live_dir else None,
        reference_data_dir=args.reference_data_dir.resolve() if args.reference_data_dir else None,
        eodhd_price_seed_path=(
            args.eodhd_price_seed_path.resolve()
            if args.eodhd_price_seed_path
            else None
        ),
        user_agent=args.user_agent,
        simfin_api_key=args.simfin_api_key,
        price_lookback_days=args.price_lookback_days,
        financial_lookback_years=args.financial_lookback_years,
        audit_years=tuple(args.audit_years),
        threshold_pct=args.threshold_pct,
        source_refresh_policy=replace(
            PRODUCTION_SOURCE_REFRESH_POLICY,
            allow_historical_revisions=args.allow_historical_revisions,
            allow_historical_price_revisions=args.allow_historical_price_revisions,
            allow_historical_price_key_removals=(
                args.allow_historical_price_key_removals
            ),
        ),
    )

    print(f"Run id: {result.run_id}")
    print(f"Mode: {result.mode}")
    print(f"Official dir: {result.live_dir}")
    print(f"Target dir: {result.target_dir}")
    print(f"Output dir: {result.output_dir}")
    print(f"Output lineage dir: {result.output_lineage_dir}")
    if result.output_snapshot_dir is not None:
        print(f"Output snapshot dir: {result.output_snapshot_dir}")
    print(f"Tickers: {result.ticker_count}")
    print(f"Price window: {result.price_start_date} -> {result.price_end_date}")
    print(f"Fallback financial years refreshed: {', '.join(str(year) for year in result.refreshed_years)}")
    print(f"SEC companyfacts years refreshed: {', '.join(str(year) for year in result.sec_companyfacts_years)}")
    print(f"Canonical price rows: {result.price_rows}")
    print(f"Canonical financial rows: {result.consolidated_rows}")
    print(f"Lineage rows: {result.lineage_rows}")
    if result.audit_dirs:
        print("Audit dirs:")
        for audit_dir in result.audit_dirs:
            print(f"  - {audit_dir}")


if __name__ == "__main__":
    main()
