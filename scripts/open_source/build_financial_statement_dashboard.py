#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from alpharank.data.open_source.financial_audit import build_financial_statement_audit_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an interactive financial statement audit dashboard.")
    parser.add_argument("--eodhd-dir", type=Path, default=Path("data/eodhd/output"))
    parser.add_argument("--open-source-dir", type=Path, default=Path("data/open_source/output"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--threshold-pct", type=float, default=1.0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "outputs" / f"financial_statement_audit_{timestamp}"
    )

    result = build_financial_statement_audit_dashboard(
        eodhd_dir=args.eodhd_dir.resolve(),
        open_source_dir=args.open_source_dir.resolve(),
        output_dir=output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        threshold_pct=args.threshold_pct,
    )

    print(f"Dashboard: {result.dashboard_path}")
    print(f"Summary markdown: {result.summary_md_path}")
    print(f"Summary json: {result.summary_json_path}")
    print(f"Alignment parquet: {result.alignment_path}")
    print(f"Issue details parquet: {result.issue_details_path}")
    print(f"Total rows: {result.total_rows}")
    print(f"Matched rows: {result.matched_rows}")
    print(f"Rows > threshold: {result.error_rows}")
    print(f"Missing in open-source: {result.missing_open_rows}")
    print(f"Extra in open-source: {result.extra_open_rows}")
    print(f"Reference rows: {result.reference_rows}")
    print(f"Ticker-quarter rows with error: {result.error_ticker_quarters}/{result.total_reference_ticker_quarters}")
    print(
        f"Ticker-quarter rows with missing in open-source: "
        f"{result.missing_open_ticker_quarters}/{result.total_reference_ticker_quarters}"
    )


if __name__ == "__main__":
    main()
