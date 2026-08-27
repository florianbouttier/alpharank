#!/usr/bin/env python3
"""Build the offline company-filterable explorer for one SEC ingestion run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from alpharank.reporting.sec_fundamental_explorer import (
    SecExplorerConfig,
    build_sec_fundamental_explorer,
)


def parse_args() -> argparse.Namespace:
    """Require one explicit RAW run and accept a deterministic output location."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-run-dir",
        type=Path,
        required=True,
        help="Explicit data/open_source/official/runs/<run_id>/raw directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to outputs/sec_fundamental_explorer/<run_id>.",
    )
    parser.add_argument(
        "--initial-ticker",
        default="AAPL.US",
        help="Company selected when the report opens.",
    )
    return parser.parse_args()


def main() -> int:
    """Build and summarize the static report without changing data pointers."""

    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    raw_run_dir = args.raw_run_dir.resolve()
    output_dir = args.output_dir or (
        project_root / "outputs" / "sec_fundamental_explorer" / raw_run_dir.parent.name
    )
    result = build_sec_fundamental_explorer(
        SecExplorerConfig(
            raw_run_dir=raw_run_dir,
            output_dir=output_dir,
            project_root=project_root,
            generated_at_utc=datetime.now(timezone.utc),
            initial_ticker=args.initial_ticker,
        )
    )
    print(
        f"SEC explorer: {result.company_count} companies, "
        f"{result.sec_row_count} SEC rows, run {result.run_id}"
    )
    print(f"Report: {result.report_path}")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
