#!/usr/bin/env python3
"""Build the versioned reference-close versus next-open return bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from alpharank.portfolio.execution import (
    apply_next_session_open_holding_returns,
    build_execution_return_bridge,
    write_execution_return_bridge,
)
from alpharank.portfolio.simulation import simulate_weighted_portfolio


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bridge(
    *,
    comparison_dir: Path,
    legacy_run_dir: Path,
    output_dir: Path,
    transaction_cost_bps: float,
    publication_blockers: tuple[str, ...] = (),
) -> Path:
    """Build one immutable paired-series report from a common replay."""

    source_paths = {
        "comparison_manifest": comparison_dir / "manifest.json",
        "canonical_holdings": comparison_dir / "comparison_common_holdings.parquet",
        "canonical_monthly": comparison_dir / "comparison_common_monthly.parquet",
        "legacy_manifest": legacy_run_dir / "data_input_manifest.json",
        "stock_prices": legacy_run_dir / "input_snapshot/US_Finalprice.parquet",
        "benchmark_prices": legacy_run_dir / "input_snapshot/SP500Price.parquet",
    }
    missing = [path for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing execution bridge inputs:\n" + "\n".join(map(str, missing)))
    if output_dir.exists():
        raise FileExistsError(f"Execution bridge output already exists: {output_dir}")

    comparison_manifest = json.loads(
        source_paths["comparison_manifest"].read_text(encoding="utf-8")
    )
    if not comparison_manifest.get("lineage_check", {}).get("passed"):
        raise ValueError("Execution bridge requires a same-snapshot common replay.")
    if comparison_manifest.get("comparison_eligible") is not True:
        raise ValueError("Execution bridge requires comparison_eligible=true.")
    declared_cost_bps = comparison_manifest.get("transaction_cost_policy", {}).get(
        "bps_times_turnover",
        comparison_manifest.get("transaction_cost_bps_times_turnover"),
    )
    if (
        declared_cost_bps is None
        or abs(float(declared_cost_bps) - float(transaction_cost_bps)) > 1e-12
    ):
        raise ValueError("Execution bridge cost rate differs from the common replay.")

    output_dir.mkdir(parents=True)

    holdings = pl.read_parquet(source_paths["canonical_holdings"])
    canonical_monthly = pl.read_parquet(source_paths["canonical_monthly"]).filter(
        pl.col("strategy") != "SPY total return"
    )
    price_columns = ["ticker", "date", "open", "close", "adjusted_close"]
    stock_prices = pl.read_parquet(source_paths["stock_prices"]).select(price_columns)
    benchmark_prices = pl.read_parquet(source_paths["benchmark_prices"]).select(price_columns)

    sensitivity_holdings = apply_next_session_open_holding_returns(
        holdings,
        stock_prices,
    )
    benchmark_tickers = benchmark_prices["ticker"].drop_nulls().unique()
    if benchmark_tickers.len() != 1:
        raise RuntimeError("Execution bridge requires exactly one benchmark ticker.")
    benchmark_seed = (
        holdings.select("decision_month", "holding_month")
        .unique()
        .with_columns(
            pl.lit("SPY next-open sensitivity").alias("strategy"),
            pl.lit(str(benchmark_tickers.item())).alias("ticker"),
            pl.lit(1.0).alias("target_weight"),
            pl.lit(0.0).alias("realized_return"),
            pl.lit(0.0).alias("benchmark_return"),
        )
    )
    benchmark_sensitivity = apply_next_session_open_holding_returns(
        benchmark_seed,
        benchmark_prices,
    ).select(
        "holding_month",
        pl.col("realized_return").alias("benchmark_return_next_open"),
    )
    sensitivity_holdings = (
        sensitivity_holdings.drop("benchmark_return")
        .join(
            benchmark_sensitivity,
            on="holding_month",
            how="left",
            validate="m:1",
        )
        .rename({"benchmark_return_next_open": "benchmark_return"})
    )
    sensitivity_monthly = simulate_weighted_portfolio(
        sensitivity_holdings,
        transaction_cost_bps=transaction_cost_bps,
        missing_return_policy="raise",
        causal_timing_policy="require_explicit",
    )
    bridge = build_execution_return_bridge(
        canonical_holdings=holdings,
        sensitivity_holdings=sensitivity_holdings,
        canonical_monthly=canonical_monthly,
        sensitivity_monthly=sensitivity_monthly,
        transaction_cost_bps=transaction_cost_bps,
    )
    policy_manifest = write_execution_return_bridge(
        bridge,
        output_dir,
        transaction_cost_bps=transaction_cost_bps,
    )
    bridge_path = output_dir / "execution_return_bridge.parquet"
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "status": (
                    "diagnostic_blocked"
                    if publication_blockers
                    else "validated_execution_convention_bridge"
                ),
                "publication_eligible": not publication_blockers,
                "publication_blockers": list(publication_blockers),
                "execution_policy": policy_manifest,
                "calendar": {
                    "start_holding_month": str(bridge["holding_month"].min()),
                    "end_holding_month": str(bridge["holding_month"].max()),
                    "months": bridge["holding_month"].n_unique(),
                },
                "outputs": {
                    "return_bridge": {
                        "path": str(bridge_path.resolve()),
                        "sha256": _sha256(bridge_path),
                    },
                    "policy": {
                        "path": str((output_dir / "execution_return_policy.json").resolve()),
                        "sha256": _sha256(output_dir / "execution_return_policy.json"),
                    },
                },
                "sources": {
                    name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                    for name, path in source_paths.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--legacy-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--publication-blocker", action="append", default=[])
    args = parser.parse_args()
    print(
        build_bridge(
            comparison_dir=args.comparison_dir.resolve(),
            legacy_run_dir=args.legacy_run_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            transaction_cost_bps=args.transaction_cost_bps,
            publication_blockers=tuple(args.publication_blocker),
        )
    )


if __name__ == "__main__":
    main()
