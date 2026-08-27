#!/usr/bin/env python3
"""Audit one refresh from data inputs through both historical portfolios."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from alpharank.governance import capture_runtime_provenance
from alpharank.replay.refresh_drift import (
    ReplayAuditInputs,
    audit_blocked_refresh,
    audit_refresh_replay,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    parser.add_argument("--common-replay-failure")
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
    report["audit_runtime_provenance"] = _capture_audit_provenance(args, report["status"])
    report_path = args.output_dir / "refresh_replay_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Refresh replay status: {report['status']}")
    print(f"Report: {report_path.resolve()}")
    return 0 if report["status"] == "identical_historical_portfolios" else 2


def _complete_inputs(args: argparse.Namespace) -> ReplayAuditInputs:
    required = (
        "candidate_snapshot",
        "baseline_legacy",
        "candidate_legacy",
        "baseline_boosting",
        "candidate_boosting",
        "baseline_common",
        "historical_cutoff",
    )
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Complete audit is missing arguments: {', '.join(missing)}")
    if (args.candidate_common is None) == (args.common_replay_failure is None):
        raise ValueError(
            "Complete audit requires exactly one of --candidate-common or --common-replay-failure"
        )
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
        common_replay_failure=args.common_replay_failure,
    )


def _capture_audit_provenance(args: argparse.Namespace, status: str) -> dict[str, object]:
    mode = (
        "blocked_refresh"
        if args.failed_refresh_run
        else "complete_replay_common_blocked"
        if args.common_replay_failure
        else "complete_replay"
    )
    return capture_runtime_provenance(
        project_root=PROJECT_ROOT,
        entrypoint="scripts/validation/audit_refresh_replay.py",
        command_argv=(sys.executable, *sys.argv),
        resolved_config={
            "mode": mode,
            "historical_cutoff": (
                args.historical_cutoff.isoformat() if args.historical_cutoff else None
            ),
            "materiality_tolerance": 1e-12,
        },
        seeds={"comparison": "deterministic_no_randomness"},
        critical_files=(
            "scripts/validation/audit_refresh_replay.py",
            "src/alpharank/replay/refresh_compare.py",
            "src/alpharank/replay/refresh_drift.py",
            "src/alpharank/replay/refresh_provenance.py",
            "src/alpharank/replay/refresh_sources.py",
        ),
        data_identifiers={
            "status": status,
            "baseline_snapshot": str(args.baseline_snapshot.resolve()),
            "candidate_snapshot": (
                str(args.candidate_snapshot.resolve()) if args.candidate_snapshot else None
            ),
            "failed_refresh_run": (
                str(args.failed_refresh_run.resolve()) if args.failed_refresh_run else None
            ),
        },
        patch_path=args.output_dir / "runtime_git_patch.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
