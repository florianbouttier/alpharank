#!/usr/bin/env python3
"""Create or enforce the versioned differential Ruff baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.quality.ruff_baseline import (
    DEFAULT_SCOPE,
    build_ruff_baseline,
    compare_ruff_baseline,
    load_baseline,
    run_ruff,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = PROJECT_ROOT / "configs/quality/ruff_baseline_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--ruff-executable", default="ruff")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    diagnostics, version = run_ruff(
        args.root,
        ruff_executable=args.ruff_executable,
        scope=DEFAULT_SCOPE,
    )
    current = build_ruff_baseline(
        args.root,
        diagnostics,
        ruff_version=version,
        scope=DEFAULT_SCOPE,
    )
    if args.write_baseline:
        write_json(args.baseline, current)
        print(args.baseline.resolve())
        return

    report = compare_ruff_baseline(load_baseline(args.baseline), current)
    if args.report is not None:
        write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
