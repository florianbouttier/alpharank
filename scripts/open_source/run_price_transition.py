#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from alpharank.data.open_source import run_open_source_price_transition


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize and audit open-source price data against EODHD reference data.")
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker roots without .US suffix.")
    parser.add_argument("--threshold-pct", type=float, default=0.5)
    parser.add_argument("--reference-data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    reference_data_dir = args.reference_data_dir or project_root / "data"
    output_dir = args.output_dir or project_root / "data" / "open_source" / f"price_transition_{args.start_date.replace('-', '')}"

    result = run_open_source_price_transition(
        reference_data_dir=reference_data_dir,
        output_dir=output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        tickers=args.tickers,
        threshold_pct=args.threshold_pct,
    )

    print(f"Open-source price transition written to: {result.output_dir}")
    print(f"Date range: {result.start_date} -> {result.end_date}")
    print(f"Tickers: {result.ticker_count}")
    print(f"Yahoo price rows: {result.yahoo_price_rows}")
    print(f"EODHD reference rows: {result.eodhd_price_rows}")
    print(f"Price alignment rows: {result.price_alignment_rows}")


if __name__ == "__main__":
    main()
