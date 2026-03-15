#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from alpharank.data.open_source import run_open_source_cadrage


def main(
    *,
    year: int = 2025,
    tickers: tuple[str, ...] | list[str] | None = None,
    universe: str = "pilot",
    threshold_pct: float = 0.5,
    reference_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    simfin_api_key: str | None = None,
    user_agent: str = "Florian Bouttier florianbouttier@example.com",
) -> None:
    result = run_open_source_cadrage(
        year=year,
        tickers=tickers,
        universe=universe,
        threshold_pct=threshold_pct,
        reference_data_dir=Path(reference_data_dir).expanduser().resolve() if reference_data_dir else None,
        output_dir=Path(output_dir).expanduser().resolve() if output_dir else None,
        user_agent=user_agent,
        simfin_api_key=simfin_api_key,
    )

    print(f"Open-source cadrage written to: {result.output_dir}")
    print(f"Tickers: {', '.join(result.tickers)}")
    print(f"S&P 500 {year} tickers audited: {result.sp500_ticker_count}")
    print(f"Tickers available in Yahoo or SEC: {result.coverage_available_in_yahoo_or_sec}")
    print(f"Yahoo price rows: {result.price_rows}")
    print(f"SEC financial rows: {result.sec_rows}")
    print(f"SEC filing financial rows: {result.sec_filing_rows}")
    print(f"SimFin financial rows: {result.simfin_rows}")
    print(f"Yahoo financial fallback tickers: {result.yfinance_financial_ticker_count}")
    print(f"Open-source consolidated financial rows: {result.consolidated_rows}")
    print(f"Open-source lineage rows: {result.lineage_rows}")
    print(f"Yahoo financial rows: {result.yfinance_financial_rows}")
    print(f"Yahoo earnings rows: {result.yfinance_earnings_rows}")
    print(f"Earnings alignment rows: {result.earnings_alignment_rows}")
    print(f"Price alignment rows: {result.price_alignment_rows}")
    print(f"Financial alignment rows: {result.financial_alignment_rows}")


if __name__ == "__main__":
    main()
