#!/usr/bin/env python3
"""Build a fail-closed Legacy/boosting/SPY replay on one common calendar."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alpharank.replay.common_strategy import (
    CommonStrategyReplayConfig,
    build_common_strategy_replay,
)
from alpharank.replay.complementary_allocation import ComplementarySleeveConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_comparison(
    *,
    legacy_run_dir: Path,
    boosting_run_dir: Path,
    output_dir: Path,
    transaction_cost_bps: float,
    top_n_values: tuple[int, ...] = (5, 10, 15, 20),
    include_legacy_valuation_universe: bool = True,
    include_causal_trend_universe: bool = False,
    complementary_sleeve: ComplementarySleeveConfig | None = None,
) -> Path:
    """Compatibility entrypoint for the maintained replay builder."""

    return build_common_strategy_replay(
        CommonStrategyReplayConfig(
            legacy_run_dir=legacy_run_dir,
            boosting_run_dir=boosting_run_dir,
            output_dir=output_dir,
            project_root=PROJECT_ROOT,
            command_argv=(sys.executable, *sys.argv),
            transaction_cost_bps=transaction_cost_bps,
            top_n_values=top_n_values,
            include_legacy_valuation_universe=include_legacy_valuation_universe,
            include_causal_trend_universe=include_causal_trend_universe,
            complementary_sleeve=complementary_sleeve,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-run-dir", type=Path, required=True)
    parser.add_argument("--boosting-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--top-n", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--include-causal-trend-universe", action="store_true")
    parser.add_argument("--complementary-top-n", type=int, help="Enable the research sleeve.")
    parser.add_argument("--complementary-max-per-sector", type=int)
    parser.add_argument("--complementary-sector-history", type=Path)
    args = parser.parse_args()
    complementary_sleeve = None
    if args.complementary_top_n is not None:
        try:
            complementary_sleeve = ComplementarySleeveConfig(
                top_n=args.complementary_top_n,
                maximum_names_per_sector=args.complementary_max_per_sector,
                sector_history_path=args.complementary_sector_history,
            )
        except ValueError as error:
            parser.error(str(error))
    elif args.complementary_max_per_sector is not None or args.complementary_sector_history:
        parser.error("Complementary sector options require --complementary-top-n.")
    print(
        build_comparison(
            legacy_run_dir=args.legacy_run_dir.resolve(),
            boosting_run_dir=args.boosting_run_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            transaction_cost_bps=args.transaction_cost_bps,
            top_n_values=tuple(args.top_n),
            include_legacy_valuation_universe=not args.native_only,
            include_causal_trend_universe=args.include_causal_trend_universe,
            complementary_sleeve=complementary_sleeve,
        )
    )


if __name__ == "__main__":
    main()
