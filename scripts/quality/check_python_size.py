#!/usr/bin/env python3
"""Create or enforce the differential Python size and complexity baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alpharank.quality.python_size import (
    build_python_size_baseline,
    compare_python_size_baselines,
    load_python_size_baseline,
    write_python_size_baseline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = PROJECT_ROOT / "configs/quality/python_size_baseline_v1.json"
DEFAULT_RUFF = str(Path(sys.executable).with_name("ruff"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--ruff-executable", default=DEFAULT_RUFF)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    current = build_python_size_baseline(
        args.root,
        ruff_executable=args.ruff_executable,
    )
    if args.write_baseline:
        write_python_size_baseline(args.baseline, current)
        print(args.baseline.resolve())
        return
    report = compare_python_size_baselines(
        load_python_size_baseline(args.baseline),
        current,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
