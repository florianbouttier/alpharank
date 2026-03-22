#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from alpharank.data.open_source import run_open_source_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified bootstrap/daily/audit ingestion for open-source market data.")
    parser.add_argument("--mode", choices=("bootstrap", "daily"), default="daily")
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker roots without .US suffix.")
    parser.add_argument("--live-dir", type=Path, default=None)
    parser.add_argument("--reference-data-dir", type=Path, default=None)
    parser.add_argument("--price-lookback-days", type=int, default=7)
    parser.add_argument("--financial-lookback-years", type=int, default=2)
    parser.add_argument("--audit-years", nargs="*", type=int, default=())
    parser.add_argument("--threshold-pct", type=float, default=0.5)
    parser.add_argument("--simfin-api-key", default=None)
    parser.add_argument("--user-agent", default="Florian Bouttier florianbouttier@example.com")
    args = parser.parse_args()

    result = run_open_source_ingestion(
        mode=args.mode,
        start_date=args.start_date,
        end_date=args.end_date,
        tickers=args.tickers,
        live_dir=args.live_dir.resolve() if args.live_dir else None,
        reference_data_dir=args.reference_data_dir.resolve() if args.reference_data_dir else None,
        user_agent=args.user_agent,
        simfin_api_key=args.simfin_api_key,
        price_lookback_days=args.price_lookback_days,
        financial_lookback_years=args.financial_lookback_years,
        audit_years=tuple(args.audit_years),
        threshold_pct=args.threshold_pct,
    )

    print(f"Run id: {result.run_id}")
    print(f"Mode: {result.mode}")
    print(f"Official dir: {result.live_dir}")
    print(f"Target dir: {result.target_dir}")
    print(f"Tickers: {result.ticker_count}")
    print(f"Price window: {result.price_start_date} -> {result.price_end_date}")
    print(f"Financial years refreshed: {', '.join(str(year) for year in result.refreshed_years)}")
    print(f"Canonical price rows: {result.price_rows}")
    print(f"Canonical financial rows: {result.consolidated_rows}")
    print(f"Lineage rows: {result.lineage_rows}")
    if result.audit_dirs:
        print("Audit dirs:")
        for audit_dir in result.audit_dirs:
            print(f"  - {audit_dir}")


if __name__ == "__main__":
    main()
