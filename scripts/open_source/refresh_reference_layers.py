#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from alpharank.data.open_source import refresh_open_source_reference_layers


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Edit these values directly in Python when needed.
START_YEAR = 2005
END_YEAR = None
AUDIT_YEARS = (2025,)
THRESHOLD_PCT = 0.5
USER_AGENT = "Florian Bouttier florianbouttier@example.com"
OFFICIAL_DIR = PROJECT_ROOT / "data" / "open_source" / "official"
REFERENCE_DATA_DIR = PROJECT_ROOT / "data"
TICKERS: tuple[str, ...] | None = None


def main() -> None:
    result = refresh_open_source_reference_layers(
        start_year=START_YEAR,
        end_year=END_YEAR,
        tickers=TICKERS,
        live_dir=OFFICIAL_DIR,
        reference_data_dir=REFERENCE_DATA_DIR,
        user_agent=USER_AGENT,
        audit_years=AUDIT_YEARS,
        threshold_pct=THRESHOLD_PCT,
    )
    print(f"Run id: {result.run_id}")
    print(f"Official dir: {result.live_dir}")
    print(f"Target dir: {result.target_dir}")
    print(f"Output dir: {result.output_dir}")
    print(f"Output lineage dir: {result.output_lineage_dir}")
    if result.output_snapshot_dir is not None:
        print(f"Output snapshot dir: {result.output_snapshot_dir}")
    print(f"Tickers: {result.ticker_count}")
    print(f"Refreshed years: {', '.join(str(year) for year in result.refreshed_years)}")
    print(f"General rows: {result.general_rows}")
    print(f"General sector non-null rows: {result.general_sector_non_null_rows}")
    print(f"Earnings rows: {result.earnings_rows}")
    print(f"Earnings tickers: {result.earnings_tickers}")
    if result.audit_dirs:
        print("Audit dirs:")
        for audit_dir in result.audit_dirs:
            print(f"  - {audit_dir}")


if __name__ == "__main__":
    main()
