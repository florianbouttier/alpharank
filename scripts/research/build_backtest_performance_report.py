#!/usr/bin/env python3
"""Build the canonical interactive performance report from one explicit replay."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from alpharank.governance_contracts.common import sha256_path
from alpharank.replay.legacy import simulate_legacy_equal_on_common_calendar
from alpharank.reporting.performance_report import (
    PerformanceReportInputs,
    SectorReportInputs,
    write_performance_report,
)

LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-replay-dir", type=Path, required=True)
    parser.add_argument("--legacy-run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--portfolio-as-of-evidence", type=Path, required=True)
    parser.add_argument("--sector-backtest-dir", type=Path, help="Sealed sector backtest package")
    parser.add_argument("--sector-history-dir", type=Path, help="Matching DEF sector release")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-frequency", choices=["daily", "monthly_archive"], default="daily"
    )
    return parser


def _write_aligned_legacy_equal(common_dir: Path, legacy_dir: Path, output_dir: Path) -> Path:
    manifest = json.loads((common_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        sha256_path(legacy_dir / "data_input_manifest.json")
        != manifest["sources"]["legacy_run_manifest"]["sha256"]
    ):
        raise ValueError("Legacy Equal must use holdings from the sealed common replay run.")
    monthly_path = common_dir / "comparison_common_monthly.parquet"
    if sha256_path(monthly_path) != manifest["artifacts"]["monthly_parquet"]["sha256"]:
        raise ValueError("Common monthly Parquet hash differs from the manifest.")
    monthly = simulate_legacy_equal_on_common_calendar(
        pl.read_parquet(legacy_dir / "legacy_common_holdings.parquet"),
        pl.read_parquet(monthly_path),
        transaction_cost_bps=manifest["transaction_cost_policy"]["bps_times_turnover"],
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "legacy_equal_common_monthly.parquet"
    monthly.write_parquet(path)
    return path


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if (args.sector_backtest_dir is None) != (args.sector_history_dir is None):
        parser.error("--sector-backtest-dir and --sector-history-dir must be supplied together")
    output_dir = args.output_dir.resolve()
    sector = (
        SectorReportInputs(
            backtest_dir=args.sector_backtest_dir.resolve(),
            sector_history_dir=args.sector_history_dir.resolve(),
            repository_root=Path(__file__).resolve().parents[2],
            legacy_equal_monthly=_write_aligned_legacy_equal(
                args.common_replay_dir.resolve(), args.legacy_run_dir.resolve(), output_dir
            ),
        )
        if args.sector_backtest_dir is not None
        else None
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = write_performance_report(
        PerformanceReportInputs(
            common_replay_dir=args.common_replay_dir.resolve(),
            legacy_run_dir=args.legacy_run_dir.resolve(),
            snapshot_manifest=args.snapshot_manifest.resolve(),
            portfolio_as_of_evidence=args.portfolio_as_of_evidence.resolve(),
            sector=sector,
        ),
        output_dir=output_dir,
        generated_at_utc=datetime.now(timezone.utc),
        analysis_frequency=args.analysis_frequency,
    )
    LOGGER.info("Performance report: %s", paths["report"])
    LOGGER.info("Manifest: %s", paths["manifest"])


if __name__ == "__main__":
    main()
