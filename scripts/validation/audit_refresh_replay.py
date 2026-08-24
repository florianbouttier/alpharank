#!/usr/bin/env python3
"""Audit one refresh from data inputs through both historical portfolios."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from alpharank.replay.refresh_drift import (
    ReplayAuditInputs,
    audit_blocked_refresh,
    audit_refresh_replay,
)


def parse_args() -> argparse.Namespace:
    """Parse either a complete comparison or a failed-refresh audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failed-refresh-run", type=Path)
    parser.add_argument("--candidate-snapshot", type=Path)
    parser.add_argument("--baseline-legacy", type=Path)
    parser.add_argument("--candidate-legacy", type=Path)
    parser.add_argument("--baseline-boosting", type=Path)
    parser.add_argument("--candidate-boosting", type=Path)
    parser.add_argument("--baseline-common", type=Path)
    parser.add_argument("--candidate-common", type=Path)
    parser.add_argument("--historical-cutoff", type=date.fromisoformat)
    return parser.parse_args()


def main() -> int:
    """Write the canonical report and fail closed on every non-identical result."""

    args = parse_args()
    if args.failed_refresh_run:
        report = audit_blocked_refresh(
            args.failed_refresh_run,
            args.baseline_snapshot,
            args.output_dir,
        )
    else:
        report = audit_refresh_replay(_complete_inputs(args), args.output_dir)
    print(f"Refresh replay status: {report['status']}")
    print(f"Report: {(args.output_dir / 'refresh_replay_report.json').resolve()}")
    return 0 if report["status"] == "identical_historical_portfolios" else 2


def _complete_inputs(args: argparse.Namespace) -> ReplayAuditInputs:
    required = (
        "candidate_snapshot",
        "baseline_legacy",
        "candidate_legacy",
        "baseline_boosting",
        "candidate_boosting",
        "baseline_common",
        "candidate_common",
        "historical_cutoff",
    )
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Complete audit is missing arguments: {', '.join(missing)}")
    return ReplayAuditInputs(
        baseline_snapshot=args.baseline_snapshot,
        candidate_snapshot=args.candidate_snapshot,
        baseline_legacy=args.baseline_legacy,
        candidate_legacy=args.candidate_legacy,
        baseline_boosting=args.baseline_boosting,
        candidate_boosting=args.candidate_boosting,
        baseline_common=args.baseline_common,
        candidate_common=args.candidate_common,
        historical_cutoff=args.historical_cutoff,
    )


if __name__ == "__main__":
    raise SystemExit(main())
