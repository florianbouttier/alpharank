#!/usr/bin/env python3
"""Build the causal Legacy/Boosting/SPY replay on one common contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.portfolio.contracts import REBALANCE_OPEN_POLICY_ID
from alpharank.portfolio.execution import LEGACY_NEXT_SESSION_OPEN
from alpharank.replay import build_common_v2_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-run-dir", type=Path, required=True)
    parser.add_argument("--boosting-run-dir", type=Path, required=True)
    parser.add_argument("--causal-snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--rank-buffer", type=int)
    parser.add_argument(
        "--holding-return-policy",
        choices=(LEGACY_NEXT_SESSION_OPEN.identifier, REBALANCE_OPEN_POLICY_ID),
        default=LEGACY_NEXT_SESSION_OPEN.identifier,
    )
    args = parser.parse_args()
    report = build_common_v2_comparison(
        legacy_run_dir=args.legacy_run_dir,
        boosting_run_dir=args.boosting_run_dir,
        causal_snapshot_dir=args.causal_snapshot_dir,
        output_dir=args.output_dir,
        top_n_values=tuple(args.top_n),
        rank_buffer=args.rank_buffer,
        holding_return_policy=args.holding_return_policy,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
