#!/usr/bin/env python3
"""Rebuild the complete React report from one explicitly selected archived report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from alpharank.reporting.strategy_analysis.dataset import read_performance_payload
from alpharank.reporting.strategy_analysis.report import write_daily_analysis_report
from alpharank.reporting.strategy_analysis.version_catalog import write_analysis_collection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-report", type=Path, action="append", default=[])
    parser.add_argument("--valuation-policy", choices=["observed_only", "last_observed_mark_v1"])
    args = parser.parse_args()
    if args.archive_report:
        paths = write_analysis_collection(
            args.source_report,
            archive_reports=args.archive_report,
            output_dir=args.output_dir,
            generated_at_utc=datetime.now(timezone.utc),
            valuation_policy=args.valuation_policy,
        )
    else:
        paths = write_daily_analysis_report(
            read_performance_payload(args.source_report),
            output_dir=args.output_dir,
            generated_at_utc=datetime.now(timezone.utc),
            source_report=args.source_report,
            valuation_policy=args.valuation_policy,
        )
    print("Rapport HTML complet :", paths["report"])
    print("Manifeste journalier :", paths["analysis_manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
