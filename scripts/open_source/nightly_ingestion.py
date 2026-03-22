#!/usr/bin/env python3
from __future__ import annotations

from datetime import date
from pathlib import Path
from datetime import datetime

from alpharank.data.open_source import run_open_source_ingestion
from alpharank.data.open_source.benchmark import load_sp500_tickers_for_year


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Edit these values directly in Python.
START_DATE = "2005-01-01"
LIVE_DIR = PROJECT_ROOT / "data" / "open_source" / "live"
REFERENCE_DATA_DIR = PROJECT_ROOT / "data"
AUDIT_YEARS = (2025,)
THRESHOLD_PCT = 0.5
PRICE_LOOKBACK_DAYS = 7
FINANCIAL_LOOKBACK_YEARS = 2
USER_AGENT = "Florian Bouttier florianbouttier@example.com"
TICKERS: tuple[str, ...] | None = None


def default_nightly_tickers() -> tuple[str, ...]:
    current_year = date.today().year
    return load_sp500_tickers_for_year(REFERENCE_DATA_DIR, current_year)


def main() -> None:
    active_tickers = TICKERS or default_nightly_tickers()
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Starting nightly ingestion")
    print(f"Universe tickers: {len(active_tickers)}")
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
    print(f"Live dir: {result.live_dir}")
    print(f"Tickers: {result.ticker_count}")
    print(f"Price window: {result.price_start_date} -> {result.price_end_date}")
    print(f"Financial years refreshed: {', '.join(str(year) for year in result.refreshed_years)}")
    if result.audit_dirs:
        print("Audit dirs:")
        for audit_dir in result.audit_dirs:
            print(f"  - {audit_dir}")


if __name__ == "__main__":
    main()
