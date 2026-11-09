#!/usr/bin/env python3
"""Replay causal sector-constrained Boosting portfolios on the full calendar."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import polars as pl

from alpharank.replay.sector_backtest import build_sector_backtest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-holdings", type=Path, required=True)
    parser.add_argument("--trend-holdings", type=Path, required=True)
    parser.add_argument("--reference-monthly", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    result = build_sector_backtest(
        pl.read_parquet(args.native_holdings),
        pl.read_parquet(args.trend_holdings),
        pl.read_parquet(args.reference_monthly),
        transaction_cost_bps=args.transaction_cost_bps,
    )
    args.output_dir.mkdir(parents=True)
    outputs = {
        "sector_holdings.parquet": result.holdings,
        "monthly_returns.parquet": result.monthly_returns,
        "performance_summary.parquet": result.performance_summary,
        "annual_returns.parquet": result.annual_returns,
        "start_year_metrics.parquet": result.start_year_metrics,
    }
    output_files: list[Path] = []
    for name, frame in outputs.items():
        parquet_path = args.output_dir / name
        csv_path = args.output_dir / name.replace(".parquet", ".csv")
        frame.write_parquet(parquet_path)
        frame.write_csv(csv_path)
        output_files.extend([parquet_path, csv_path])

    input_paths = (args.native_holdings, args.trend_holdings, args.reference_monthly)
    input_manifests = [path.parent / "manifest.json" for path in input_paths]

    manifest = {
        "contract_version": 1,
        "status": "validated_r_and_d",
        "promotion_eligible": False,
        "methodology": {
            "allocation": "Boosting Top 5 equal weight, at most one ticker per causal SEC sector",
            "transaction_cost_bps_times_turnover": args.transaction_cost_bps,
            "benchmark": "SPY total return from adjusted_close",
            "missing_selected_return_policy": "raise",
            "score_only_tail_policy": "trailing horizon_pending month excluded",
            "prediction_timing_compatibility": "legacy_month_only for immutable audited predictions",
        },
        "calendar": {
            "months": result.monthly_returns.filter(
                pl.col("strategy") == "Boosting Top 5 | 1 per sector"
            ).height,
            "start_holding_month": str(result.monthly_returns["holding_month"].min()),
            "end_holding_month": str(result.monthly_returns["holding_month"].max()),
        },
        "inputs": {str(path): _sha256(path) for path in input_paths},
        "input_manifests": {str(path): _sha256(path) for path in input_manifests if path.exists()},
        "files": {path.name: _sha256(path) for path in output_files},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logging.info("output=%s", args.output_dir)
    logging.info("calendar=%s", manifest["calendar"])
    print(
        result.performance_summary.select(
            "strategy", "cagr", "annualized_volatility", "max_drawdown", "years_beating_spy"
        )
    )
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
