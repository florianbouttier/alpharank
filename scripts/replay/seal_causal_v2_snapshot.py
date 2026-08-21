#!/usr/bin/env python3
"""Seal one production-composed AlphaRank snapshot for the causal-v2 replay."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from alpharank.replay import (
    seal_causal_v2_snapshot,
    validate_causal_v2_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    args = _parse_args()
    if args.validate_only:
        result = validate_causal_v2_snapshot(args.output_dir)
    else:
        commit = args.implementation_commit or subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        result = seal_causal_v2_snapshot(
            source_snapshot_dir=args.source_snapshot_dir,
            package_dir=args.output_dir,
            project_root=PROJECT_ROOT,
            command_argv=[sys.executable, *sys.argv],
            implementation_commit=commit,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-snapshot-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and args.source_snapshot_dir is None:
        parser.error("--source-snapshot-dir is required unless --validate-only is used")
    return args


if __name__ == "__main__":
    main()
