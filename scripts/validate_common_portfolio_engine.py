#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl

from alpharank.portfolio.adapters.legacy import legacy_detailed_to_holdings
from alpharank.portfolio.lineage import (
    compare_input_hashes,
    input_hashes_from_manifest,
    load_manifest,
)
from alpharank.portfolio.simulation import simulate_weighted_portfolio
from alpharank.strategy.legacy import StrategyLearner
from alpharank.utils.frame_backend import to_pandas


def _maximum_absolute_error(frame: pl.DataFrame, left: str, right: str) -> float:
    return float((frame[left] - frame[right]).abs().max()) if frame.height else float("inf")


def validate_legacy(
    detailed_path: Path,
    aggregated_path: Path,
    *,
    tolerance: float,
) -> dict[str, Any]:
    detailed = pl.read_parquet(detailed_path)
    aggregated = pl.read_parquet(aggregated_path)
    benchmark = aggregated.filter(pl.col("portfolio_model") == "SP500").select(
        "year_month",
        "monthly_return",
    )
    rows: list[dict[str, Any]] = []
    for portfolio_model in ("Combined_Equal", "Combined_Frequency"):
        source = detailed.filter(pl.col("portfolio_model") == portfolio_model)
        expected = (
            aggregated.filter(
                (pl.col("portfolio_model") == portfolio_model)
                & pl.col("monthly_return").is_not_null()
            )
            .select(
                pl.col("year_month").cast(pl.Date).alias("holding_month"),
                pl.col("monthly_return").alias("expected_return"),
            )
            .unique("holding_month")
        )
        holdings = legacy_detailed_to_holdings(
            source,
            strategy=portfolio_model,
            benchmark_monthly=benchmark,
        ).filter(pl.col("benchmark_return").is_not_null())
        replay = simulate_weighted_portfolio(
            holdings,
            transaction_cost_bps=0.0,
            # This validator reproduces the frozen historical Legacy baseline.
            # New simulations fail closed on missing selected returns by default.
            missing_return_policy="renormalize_available",
        )
        joined = replay.join(expected, on="holding_month", how="inner")
        maximum_error = _maximum_absolute_error(joined, "net_return", "expected_return")
        rows.append(
            {
                "portfolio_model": portfolio_model,
                "holding_month_start": str(joined["holding_month"].min()),
                "holding_month_end": str(joined["holding_month"].max()),
                "months": joined.height,
                "expected_complete_months": replay.height,
                "maximum_absolute_return_error": maximum_error,
                "passed": joined.height == replay.height and maximum_error <= tolerance,
            }
        )
    source_labels = (
        "Legacy_Optuna_11",
        "Legacy_Optuna_12",
        "Legacy_Optuna_21",
        "Legacy_Optuna_22",
    )
    source_outputs = [
        {
            "detailed": to_pandas(
                detailed.filter(pl.col("portfolio_model") == label).drop("portfolio_model")
            )
        }
        for label in source_labels
    ]
    index = SimpleNamespace(monthly_returns=to_pandas(benchmark))
    aggregation_rows: list[dict[str, Any]] = []
    for mode, portfolio_model in (
        ("equal", "Combined_Equal"),
        ("frequency", "Combined_Frequency"),
    ):
        rebuilt = StrategyLearner.aggregate_portfolios(
            source_outputs,
            mode=mode,
            index=index,
            backend="polars",
        )["aggregated"][["year_month", "monthly_return"]].copy()
        rebuilt["year_month"] = rebuilt["year_month"].astype(str)
        expected_pd = to_pandas(
            aggregated.filter(pl.col("portfolio_model") == portfolio_model).select(
                "year_month",
                pl.col("monthly_return").alias("expected_return"),
            )
        )
        expected_pd["year_month"] = expected_pd["year_month"].dt.to_period("M").astype(str)
        joined_pd = rebuilt.merge(expected_pd, on="year_month", how="inner")
        maximum_error = float(
            (joined_pd["monthly_return"] - joined_pd["expected_return"]).abs().max()
        )
        aggregation_rows.append(
            {
                "portfolio_model": portfolio_model,
                "months": len(joined_pd),
                "maximum_absolute_return_error": maximum_error,
                "passed": len(joined_pd) == len(rebuilt) and maximum_error <= tolerance,
            }
        )
    return {
        "detailed_path": str(detailed_path.resolve()),
        "aggregated_path": str(aggregated_path.resolve()),
        "portfolios": rows,
        "strategy_aggregation_path": aggregation_rows,
        "passed": all(row["passed"] for row in rows + aggregation_rows),
    }


def validate_alpha(
    holdings_path: Path,
    monthly_path: Path,
    *,
    transaction_cost_bps: float,
    tolerance: float,
) -> dict[str, Any]:
    raw_holdings = pl.read_parquet(holdings_path)
    holdings = raw_holdings.with_columns(
        pl.col("portfolio_weight").alias("target_weight"),
        pl.col("future_return_1m").alias("realized_return"),
        pl.col("benchmark_future_return_1m").alias("benchmark_return"),
    ).select(
        "strategy",
        "decision_month",
        "holding_month",
        "ticker",
        "target_weight",
        "realized_return",
        "benchmark_return",
        "sector",
    )
    replay = simulate_weighted_portfolio(
        holdings,
        transaction_cost_bps=transaction_cost_bps,
    )
    expected = pl.read_csv(monthly_path, try_parse_dates=True).select(
        "strategy",
        "holding_month",
        pl.col("gross_return").alias("expected_gross_return"),
        pl.col("net_return").alias("expected_net_return"),
        pl.col("turnover").alias("expected_turnover"),
        pl.col("benchmark_return").alias("expected_benchmark_return"),
    )
    joined = replay.join(expected, on=["strategy", "holding_month"], how="inner")
    errors = {
        "gross_return": _maximum_absolute_error(joined, "gross_return", "expected_gross_return"),
        "net_return": _maximum_absolute_error(joined, "net_return", "expected_net_return"),
        "turnover": _maximum_absolute_error(joined, "turnover", "expected_turnover"),
        "benchmark_return": _maximum_absolute_error(
            joined,
            "benchmark_return",
            "expected_benchmark_return",
        ),
    }
    return {
        "holdings_path": str(holdings_path.resolve()),
        "monthly_path": str(monthly_path.resolve()),
        "strategies": replay["strategy"].n_unique(),
        "months": replay.height,
        "joined_rows": joined.height,
        "transaction_cost_bps": transaction_cost_bps,
        "maximum_absolute_errors": errors,
        "passed": joined.height == replay.height and all(value <= tolerance for value in errors.values()),
    }


def validate_data_lineage(
    legacy_manifest_path: Path,
    alpha_manifest_path: Path,
) -> dict[str, Any]:
    legacy_hashes = input_hashes_from_manifest(load_manifest(legacy_manifest_path))
    alpha_hashes = input_hashes_from_manifest(load_manifest(alpha_manifest_path))
    report = compare_input_hashes(
        legacy_hashes,
        alpha_hashes,
        required_keys=set(alpha_hashes),
    )
    report.update(
        {
            "legacy_manifest": str(legacy_manifest_path.resolve()),
            "alpha_manifest": str(alpha_manifest_path.resolve()),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay frozen Legacy and Alpha portfolios through the shared portfolio engine."
    )
    parser.add_argument("--legacy-detailed", type=Path, required=True)
    parser.add_argument("--legacy-aggregated", type=Path, required=True)
    parser.add_argument("--alpha-holdings", type=Path, required=True)
    parser.add_argument("--alpha-monthly", type=Path, required=True)
    parser.add_argument("--legacy-data-manifest", type=Path)
    parser.add_argument("--alpha-data-manifest", type=Path)
    parser.add_argument(
        "--allow-distinct-snapshots",
        action="store_true",
        help=(
            "Allow mechanical parity validation to finish when data lineage is "
            "missing or different. The report remains comparison_eligible=false."
        ),
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifests = (args.legacy_data_manifest, args.alpha_data_manifest)
    if any(manifests) and not all(manifests):
        parser.error(
            "--legacy-data-manifest and --alpha-data-manifest must be provided together"
        )
    if not all(manifests) and not args.allow_distinct_snapshots:
        parser.error(
            "data manifests are required; use --allow-distinct-snapshots only for "
            "mechanical replay checks that must not be used as performance comparisons"
        )

    report = {
        "tolerance": args.tolerance,
        "legacy": validate_legacy(
            args.legacy_detailed,
            args.legacy_aggregated,
            tolerance=args.tolerance,
        ),
        "alpha": validate_alpha(
            args.alpha_holdings,
            args.alpha_monthly,
            transaction_cost_bps=args.transaction_cost_bps,
            tolerance=args.tolerance,
        ),
    }
    if all(manifests):
        report["data_lineage"] = validate_data_lineage(*manifests)
    else:
        report["data_lineage"] = {
            "passed": False,
            "status": "not_checked",
            "reason": "data manifests were not supplied",
        }
    report["engine_parity_passed"] = (
        report["legacy"]["passed"] and report["alpha"]["passed"]
    )
    report["comparison_eligible"] = report["data_lineage"]["passed"]
    report["passed"] = (
        report["engine_parity_passed"] and report["comparison_eligible"]
    )
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if not report["engine_parity_passed"] or (
        not report["comparison_eligible"] and not args.allow_distinct_snapshots
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
