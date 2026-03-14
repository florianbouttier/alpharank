#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from alpharank.data.open_source import run_open_source_cadrage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 2025 open-source data cadrage pilot.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--tickers", nargs="*", default=None, help="Pilot tickers without .US suffix.")
    parser.add_argument("--universe", choices=["pilot", "sp500-2025"], default="pilot")
    parser.add_argument("--threshold-pct", type=float, default=0.5)
    parser.add_argument("--reference-data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--simfin-api-key", default=None, help="Optional SimFin API key. Falls back to SIMFIN_API_KEY env var.")
    parser.add_argument(
        "--user-agent",
        default="Florian Bouttier florianbouttier@example.com",
        help="SEC-compatible User-Agent header.",
    )
    args = parser.parse_args()

    result = run_open_source_cadrage(
        year=args.year,
        tickers=args.tickers,
        universe=args.universe,
        threshold_pct=args.threshold_pct,
        reference_data_dir=args.reference_data_dir,
        output_dir=args.output_dir,
        user_agent=args.user_agent,
        simfin_api_key=args.simfin_api_key,
    )

    print(f"Open-source cadrage written to: {result.output_dir}")
    print(f"Tickers: {', '.join(result.tickers)}")
    print(f"S&P 500 {args.year} tickers audited: {result.sp500_ticker_count}")
    print(f"Tickers available in Yahoo or SEC: {result.coverage_available_in_yahoo_or_sec}")
    print(f"Yahoo price rows: {result.price_rows}")
    print(f"SEC financial rows: {result.sec_rows}")
    print(f"SimFin financial rows: {result.simfin_rows}")
    print(f"Best-effort financial rows: {result.best_effort_rows}")
    print(f"Yahoo financial rows: {result.yfinance_financial_rows}")
    print(f"Yahoo earnings rows: {result.yfinance_earnings_rows}")
    print(f"Earnings alignment rows: {result.earnings_alignment_rows}")
    print(f"Price alignment rows: {result.price_alignment_rows}")
    print(f"Financial alignment rows: {result.financial_alignment_rows}")


if __name__ == "__main__":
    main()
