#!/usr/bin/env python3
from __future__ import annotations

from datetime import date
from datetime import datetime
from pathlib import Path

import polars as pl

from alpharank.data.open_source import run_open_source_ingestion
from alpharank.data.open_source.benchmark import load_sp500_tickers_for_year


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Edit these values directly in Python.
START_DATE = "2005-01-01"
LIVE_DIR = PROJECT_ROOT / "data" / "open_source" / "official"
REFERENCE_DATA_DIR = PROJECT_ROOT / "data"
AUDIT_YEARS = (2025,)
THRESHOLD_PCT = 0.5
PRICE_LOOKBACK_DAYS = 7
FINANCIAL_LOOKBACK_YEARS = 2
USER_AGENT = "Florian Bouttier florianbouttier@example.com"
TICKERS: tuple[str, ...] | None = None


RAW_TICKER_FILES = (
    "general_reference.parquet",
    "prices_yfinance.parquet",
    "financials_sec_companyfacts.parquet",
    "financials_sec_filing.parquet",
    "financials_simfin.parquet",
    "financials_yfinance.parquet",
    "earnings_yfinance.parquet",
)


def load_existing_live_tickers(live_dir: Path = LIVE_DIR) -> tuple[str, ...]:
    raw_dir = live_dir / "raw"
    tickers: set[str] = set()
    for file_name in RAW_TICKER_FILES:
        path = raw_dir / file_name
        if not path.exists():
            continue
        frame = pl.read_parquet(path, columns=["ticker"])
        tickers.update(
            ticker
            for ticker in (
                frame.select(pl.col("ticker").cast(pl.Utf8, strict=False).str.replace(r"\.US$", "")).to_series().to_list()
            )
            if ticker
        )
    return tuple(sorted(tickers))


def default_nightly_tickers(
    *,
    reference_data_dir: Path = REFERENCE_DATA_DIR,
    live_dir: Path = LIVE_DIR,
) -> tuple[str, ...]:
    current_year = date.today().year
    current_sp500 = set(load_sp500_tickers_for_year(reference_data_dir, current_year))
    existing_live = set(load_existing_live_tickers(live_dir))
    return tuple(sorted(current_sp500 | existing_live))


def main() -> None:
    current_sp500 = load_sp500_tickers_for_year(REFERENCE_DATA_DIR, date.today().year)
    existing_live = load_existing_live_tickers(LIVE_DIR)
    active_tickers = TICKERS or default_nightly_tickers(reference_data_dir=REFERENCE_DATA_DIR, live_dir=LIVE_DIR)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Starting nightly ingestion")
    print(f"Universe tickers: {len(active_tickers)}")
    if TICKERS is None:
        print(f"Current S&P 500 tickers: {len(current_sp500)}")
        print(f"Already tracked live tickers: {len(existing_live)}")
        print("Nightly default universe = union(current S&P 500, existing live tickers)")
    result = run_open_source_ingestion(
        mode="daily",
        start_date=START_DATE,
        tickers=active_tickers,
        live_dir=LIVE_DIR,
        reference_data_dir=REFERENCE_DATA_DIR,
        audit_years=AUDIT_YEARS,
        threshold_pct=THRESHOLD_PCT,
        price_lookback_days=PRICE_LOOKBACK_DAYS,
        financial_lookback_years=FINANCIAL_LOOKBACK_YEARS,
        user_agent=USER_AGENT,
    )
    print(f"Nightly ingestion completed: {result.run_id}")
    print(f"Official dir: {result.live_dir}")
    print(f"Target dir: {result.target_dir}")
    print(f"Output dir: {result.output_dir}")
    print(f"Output lineage dir: {result.output_lineage_dir}")
    print(f"Tickers: {result.ticker_count}")
    print(f"Price window: {result.price_start_date} -> {result.price_end_date}")
    print(f"Financial years refreshed: {', '.join(str(year) for year in result.refreshed_years)}")
    if result.audit_dirs:
        print("Audit dirs:")
        for audit_dir in result.audit_dirs:
            print(f"  - {audit_dir}")


if __name__ == "__main__":
    main()
