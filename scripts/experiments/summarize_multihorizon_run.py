#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpharank.multihorizon.metrics import score_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute a multi-horizon run's canonical metrics.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top-n", default="5,10,20")
    args = parser.parse_args()
    top_n_values = tuple(int(value) for value in args.top_n.split(","))
    summary_rows: list[dict] = []
    for prediction_path in sorted(args.run_dir.glob("*_h*/predictions.parquet")):
        combination_dir = prediction_path.parent
        method, raw_horizon = combination_dir.name.rsplit("_h", maxsplit=1)
        horizon = int(raw_horizon)
        predictions = pl.read_parquet(prediction_path).filter(
            pl.col(f"future_excess_return_{horizon}m").is_not_null()
        )
        fold_rows: list[dict] = []
        portfolio_parts: list[pl.DataFrame] = []
        for fold_frame in predictions.partition_by("fold", maintain_order=True):
            fold = int(fold_frame["fold"][0])
            metrics, portfolio = score_predictions(
                fold_frame,
                method=method,
                horizon=horizon,
                top_n_values=top_n_values,
            )
            fold_rows.append({"fold": fold, **metrics})
            portfolio_parts.append(
                portfolio.with_columns(
                    pl.lit(fold).alias("fold"),
                    pl.lit(method).alias("method"),
                    pl.lit(horizon).alias("horizon"),
                )
            )
        overall, _ = score_predictions(
            predictions,
            method=method,
            horizon=horizon,
            top_n_values=top_n_values,
        )
        pl.DataFrame(fold_rows).write_csv(combination_dir / "fold_metrics.csv")
        pl.concat(portfolio_parts).write_csv(combination_dir / "portfolio_monthly.csv")
        summary_rows.append(
            {
                "method": method,
                "horizon": horizon,
                "folds": len(fold_rows),
                "test_rows": predictions.height,
                **overall,
            }
        )
    pl.DataFrame(summary_rows).sort(["method", "horizon"]).write_csv(
        args.run_dir / "model_horizon_summary.csv"
    )


if __name__ == "__main__":
    main()
