#!/usr/bin/env python3
"""Build an explicit, reusable Legacy portfolio replay from a frozen run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpharank.portfolio.adapters.legacy import legacy_detailed_to_holdings
from alpharank.portfolio.artifacts import write_common_portfolio_artifacts
from alpharank.portfolio.benchmark import (
    SPY_TOTAL_RETURN,
    benchmark_convention,
    completed_through_month,
    monthly_benchmark_returns,
)
from alpharank.portfolio.comparison import reference_monthly_series
from alpharank.portfolio.simulation import simulate_weighted_portfolio


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_replay(
    *,
    run_dir: Path,
    output_dir: Path,
    benchmark_id: str,
) -> tuple[dict[str, Path], Path]:
    convention = benchmark_convention(benchmark_id)
    detailed_path = run_dir / "legacy_detailed_returns_polars.parquet"
    prices_path = run_dir / "input_snapshot/SP500Price.parquet"
    data_manifest_path = run_dir / "data_input_manifest.json"
    for path in (detailed_path, prices_path, data_manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    detailed = pl.read_parquet(detailed_path)
    prices = pl.read_parquet(prices_path)
    price_max_date = prices.select(
        pl.col("date").cast(pl.Date, strict=False).max()
    ).item()
    completed_through = completed_through_month(prices)
    benchmark = monthly_benchmark_returns(prices, convention=convention).filter(
        pl.col("year_month") <= completed_through
    )

    holdings_parts: list[pl.DataFrame] = []
    monthly_parts: list[pl.DataFrame] = []
    for strategy in ("Combined_Equal", "Combined_Frequency"):
        holdings = legacy_detailed_to_holdings(
            detailed.filter(pl.col("portfolio_model") == strategy),
            strategy=strategy,
            benchmark_monthly=benchmark,
        ).filter(pl.col("benchmark_return").is_not_null())
        holdings_parts.append(holdings)
        monthly_parts.append(
            simulate_weighted_portfolio(
                holdings,
                transaction_cost_bps=0.0,
                causal_timing_policy="legacy_month_only",
            )
        )
    holdings = pl.concat(holdings_parts, how="diagonal_relaxed")
    monthly = pl.concat(monthly_parts, how="diagonal_relaxed")
    monthly = pl.concat(
        [
            monthly,
            reference_monthly_series(
                monthly.filter(pl.col("strategy") == "Combined_Frequency"),
                strategy=convention.label,
                return_column="benchmark_return",
            ),
        ],
        how="diagonal_relaxed",
    )
    metadata = {
        "id": convention.identifier,
        "label": convention.label,
        "price_column": convention.price_column,
        "includes_distributions": convention.includes_distributions,
        "price_max_date": str(price_max_date),
        "completed_through_month": str(completed_through),
    }
    prefix = "legacy_common_total_return" if convention == SPY_TOTAL_RETURN else "legacy_common_price_return"
    artifacts = write_common_portfolio_artifacts(
        output_dir=output_dir,
        holdings=holdings,
        monthly_returns=monthly,
        prefix=prefix,
        benchmark_metadata=metadata,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "canonical_legacy_common_replay",
                "run_dir": str(run_dir.resolve()),
                "run_id": run_dir.name,
                "benchmark": metadata,
                "sources": {
                    "legacy_detailed": {
                        "path": str(detailed_path.resolve()),
                        "sha256": _hash(detailed_path),
                    },
                    "benchmark_prices": {
                        "path": str(prices_path.resolve()),
                        "sha256": _hash(prices_path),
                    },
                    "data_input_manifest": {
                        "path": str(data_manifest_path.resolve()),
                        "sha256": _hash(data_manifest_path),
                    },
                },
                "artifacts": {
                    name: {
                        "path": str(path.resolve()),
                        "sha256": _hash(path),
                    }
                    for name, path in artifacts.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifacts, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark-id",
        default=SPY_TOTAL_RETURN.identifier,
        choices=("spy_total_return_adjusted_close", "spy_price_return_close"),
    )
    args = parser.parse_args()
    artifacts, manifest = build_replay(
        run_dir=args.run_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        benchmark_id=args.benchmark_id,
    )
    print(artifacts["performance_csv"])
    print(manifest)


if __name__ == "__main__":
    main()
