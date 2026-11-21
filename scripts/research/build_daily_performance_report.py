#!/usr/bin/env python3
"""Rebuild the complete React report from one explicitly selected archived report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from alpharank.reporting.strategy_analysis.dataset import read_performance_payload
from alpharank.reporting.strategy_analysis.report import write_daily_analysis_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = write_daily_analysis_report(
        read_performance_payload(args.source_report),
        output_dir=args.output_dir,
        generated_at_utc=datetime.now(timezone.utc),
        source_report=args.source_report,
    )
    print("Rapport HTML complet :", paths["report"])
    print("Manifeste journalier :", paths["analysis_manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
