#!/usr/bin/env python3
"""Verify or regenerate dependency views from pyproject.toml."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.quality.dependencies import (
    dependency_sync_report,
    load_dependency_source,
    write_dependency_views,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    source = load_dependency_source(root / "pyproject.toml")
    requirements = root / "requirements.txt"
    environment = root / "environment.yml"
    if args.write:
        write_dependency_views(
            source,
            requirements_path=requirements,
            environment_path=environment,
        )
    report = dependency_sync_report(
        source,
        requirements_path=requirements,
        environment_path=environment,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
