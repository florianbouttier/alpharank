#!/usr/bin/env python3
"""Build the canonical interactive performance report from one explicit replay."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from alpharank.reporting.performance_report import (
    PerformanceReportInputs,
    write_performance_report,
)

LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-replay-dir", type=Path, required=True)
    parser.add_argument("--legacy-run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--portfolio-as-of-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = write_performance_report(
        PerformanceReportInputs(
            common_replay_dir=args.common_replay_dir.resolve(),
            legacy_run_dir=args.legacy_run_dir.resolve(),
            snapshot_manifest=args.snapshot_manifest.resolve(),
            portfolio_as_of_evidence=args.portfolio_as_of_evidence.resolve(),
        ),
        output_dir=args.output_dir.resolve(),
        generated_at_utc=datetime.now(timezone.utc),
    )
    LOGGER.info("Performance report: %s", paths["report"])
    LOGGER.info("Manifest: %s", paths["manifest"])


if __name__ == "__main__":
    main()
