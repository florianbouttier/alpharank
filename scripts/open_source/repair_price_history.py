#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from alpharank.data.open_source import repair_open_source_price_history


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Edit these values directly in Python.
START_DATE = "2005-01-01"
END_DATE = None
OFFICIAL_DIR = PROJECT_ROOT / "data" / "open_source" / "official"
REFERENCE_DATA_DIR = PROJECT_ROOT / "data"
AUDIT_YEARS = ()
THRESHOLD_PCT = 0.5
TICKERS: tuple[str, ...] | None = None


def main() -> None:
    result = repair_open_source_price_history(
        start_date=START_DATE,
        end_date=END_DATE,
        tickers=TICKERS,
        live_dir=OFFICIAL_DIR,
        reference_data_dir=REFERENCE_DATA_DIR,
        audit_years=AUDIT_YEARS,
        threshold_pct=THRESHOLD_PCT,
    )
    print(f"Run id: {result.run_id}")
    print(f"Mode: {result.mode}")
    print(f"Official dir: {result.live_dir}")
    print(f"Output dir: {result.output_dir}")
    print(f"Output lineage dir: {result.output_lineage_dir}")
    if result.output_snapshot_dir is not None:
        print(f"Output snapshot dir: {result.output_snapshot_dir}")
    print(f"Tickers considered: {result.ticker_count}")
    print(f"Price window: {result.price_start_date} -> {result.price_end_date}")
    print(f"Canonical price rows: {result.price_rows}")
    if result.audit_dirs:
        print("Audit dirs:")
        for audit_dir in result.audit_dirs:
            print(f"  - {audit_dir}")


if __name__ == "__main__":
    main()
