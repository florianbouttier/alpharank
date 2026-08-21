#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import polars as pl

from alpharank.data.contracts.ticker_integrity import load_ticker_exclusion_registry
from alpharank.multihorizon.live import (
    LiveAlphaConfig,
    previous_completed_month,
    run_live_alpha,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the frozen EMA-only h6 Alpha model and score a live completed month."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--legacy-detailed", type=Path, required=True)
    parser.add_argument("--decision-month", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    decision_month = (
        date.fromisoformat(args.decision_month)
        if args.decision_month
        else _default_decision_month(args.data_dir)
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or PROJECT_ROOT
        / "outputs"
        / "live_alpha"
        / f"ema_classification_h6_{decision_month:%Y%m}_{timestamp}"
    )
    exclusions = load_ticker_exclusion_registry().excluded_tickers
    result = run_live_alpha(
        LiveAlphaConfig(
            data_dir=args.data_dir.resolve(),
            legacy_detailed_returns_path=args.legacy_detailed.resolve(),
            output_dir=output_dir.resolve(),
            decision_month=decision_month,
            excluded_tickers=exclusions,
        )
    )
    print(f"Live Alpha run: {result}")
    print(f"HTML: {result / 'html' / 'live_alpha_portfolio.html'}")


def _default_decision_month(data_dir: Path) -> date:
    benchmark = pl.read_parquet(data_dir / "SP500Price.parquet", columns=["date"])
    latest_date = benchmark.select(pl.col("date").cast(pl.Date).max()).item()
    if latest_date is None:
        raise ValueError("SP500Price.parquet has no usable date.")
    return previous_completed_month(latest_date)


if __name__ == "__main__":
    main()
