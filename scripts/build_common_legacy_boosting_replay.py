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
    args = parser.parse_args()
    print(
        build_comparison(
            legacy_run_dir=args.legacy_run_dir.resolve(),
            boosting_run_dir=args.boosting_run_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            transaction_cost_bps=args.transaction_cost_bps,
            top_n_values=tuple(args.top_n),
            include_legacy_valuation_universe=not args.native_only,
            include_causal_trend_universe=args.include_causal_trend_universe,
        )
    )


if __name__ == "__main__":
    main()
