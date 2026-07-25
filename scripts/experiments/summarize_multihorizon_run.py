#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpharank.multihorizon.metrics import score_predictions
from alpharank.multihorizon.trading import (
    evaluate_trading_predictions,
    performance_statistics,
)


def _legacy_monthly(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .filter(pl.col("model") == "Combined_Frequency")
        .select(
            pl.col("year_month").cast(pl.Date).alias("holding_month"),
            pl.col("monthly_return").alias("legacy_return"),
        )
        .unique("holding_month")
        .sort("holding_month")
    )


def _attach_legacy_metrics(
    monthly: pl.DataFrame,
    summary: pl.DataFrame,
    legacy: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    enriched = monthly.join(legacy, on="holding_month", how="left")
    rows: list[dict] = []
    for row in summary.to_dicts():
        top_n = int(row["top_n"])
        top_monthly = enriched.filter(
            (pl.col("top_n") == top_n) & pl.col("legacy_return").is_not_null()
        )
        legacy_metrics = performance_statistics(top_monthly["legacy_return"].to_numpy())
        rows.append(
            {
                **row,
                "legacy_total_return": legacy_metrics["total_return"],
                "legacy_cagr": legacy_metrics["cagr"],
                "legacy_sharpe": legacy_metrics["sharpe"],
                "legacy_max_drawdown": legacy_metrics["max_drawdown"],
            }
        )
    return enriched, pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute a multi-horizon run's canonical metrics.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top-n", default="5,10,20")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--legacy-monthly", type=Path)
    args = parser.parse_args()
    top_n_values = tuple(int(value) for value in args.top_n.split(","))
    if args.legacy_monthly is None:
        manifest = json.loads((args.run_dir / "manifest.json").read_text())
        args.legacy_monthly = Path(manifest["config"]["legacy_monthly_returns_path"])
    legacy = _legacy_monthly(args.legacy_monthly)
    summary_rows: list[dict] = []
    coverage_rows: list[dict] = []
    prediction_frames: dict[tuple[str, int], pl.DataFrame] = {}
    trading_summary_parts: list[pl.DataFrame] = []
    for prediction_path in sorted(args.run_dir.glob("*_h*/predictions.parquet")):
        combination_dir = prediction_path.parent
        method, raw_horizon = combination_dir.name.rsplit("_h", maxsplit=1)
        horizon = int(raw_horizon)
        predictions = pl.read_parquet(prediction_path).filter(
            pl.col(f"future_excess_return_{horizon}m").is_not_null()
        )
        prediction_frames[(method, horizon)] = predictions
        coverage_rows.append(
            {
                "method": method,
                "horizon": horizon,
                "start_decision_month": predictions["decision_month"].min(),
                "end_decision_month": predictions["decision_month"].max(),
                "test_months": predictions["decision_month"].n_unique(),
                "test_rows": predictions.height,
                "outer_folds": predictions["fold"].n_unique(),
            }
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
        trading_monthly, trading_summary = evaluate_trading_predictions(
            predictions,
            top_n_values=top_n_values,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        trading_monthly, trading_summary = _attach_legacy_metrics(
            trading_monthly,
            trading_summary,
            legacy,
        )
        trading_monthly.write_csv(combination_dir / "trading_monthly.csv")
        trading_summary = trading_summary.with_columns(
            pl.lit(method).alias("method"),
            pl.lit(horizon).alias("horizon"),
            pl.lit("native").alias("comparison_period"),
        )
        trading_summary.write_csv(combination_dir / "trading_backtest.csv")
        trading_summary_parts.append(trading_summary)
    pl.DataFrame(summary_rows).sort(["method", "horizon"]).write_csv(
        args.run_dir / "model_horizon_summary.csv"
    )
    coverage = pl.DataFrame(coverage_rows).sort(["method", "horizon"])
    coverage.write_csv(args.run_dir / "test_coverage.csv")
    pl.concat(trading_summary_parts, how="diagonal_relaxed").sort(
        ["method", "horizon", "top_n"]
    ).write_csv(args.run_dir / "trading_backtest_all.csv")

    economic_coverage = coverage.filter(pl.col("method") != "teacher")
    common_start = economic_coverage["start_decision_month"].max()
    common_end = economic_coverage["end_decision_month"].min()
    common_parts: list[pl.DataFrame] = []
    for (method, horizon), predictions in prediction_frames.items():
        if method == "teacher":
            continue
        common_predictions = predictions.filter(
            pl.col("decision_month").is_between(common_start, common_end, closed="both")
        )
        monthly, summary = evaluate_trading_predictions(
            common_predictions,
            top_n_values=top_n_values,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        _, summary = _attach_legacy_metrics(monthly, summary, legacy)
        common_parts.append(
            summary.with_columns(
                pl.lit(method).alias("method"),
                pl.lit(horizon).alias("horizon"),
                pl.lit(f"{common_start}_{common_end}").alias("comparison_period"),
            )
        )
    pl.concat(common_parts, how="diagonal_relaxed").sort(
        ["method", "horizon", "top_n"]
    ).write_csv(args.run_dir / "trading_backtest_common_period.csv")


if __name__ == "__main__":
    main()
