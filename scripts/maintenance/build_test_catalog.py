#!/usr/bin/env python3
"""Build or validate the tracked Pytest file catalog from one JUnit run."""

from __future__ import annotations

import argparse
from pathlib import Path

from alpharank.quality.test_catalog import (
    build_test_catalog,
    tracked_test_paths,
    write_test_catalog,
)
from alpharank.quality.test_suites import load_test_suite_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = PROJECT_ROOT / "configs" / "quality" / "test_suites_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "architecture" / "test_catalog_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--junitxml", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--measured-at", required=True)
    parser.add_argument("--measurement-command", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    catalog = build_test_catalog(
        tracked_test_paths(root),
        load_test_suite_policy(args.policy.resolve()),
        junit_path=args.junitxml.resolve(),
        measured_at=args.measured_at,
        measurement_command=args.measurement_command,
    )
    write_test_catalog(args.output.resolve(), catalog)


if __name__ == "__main__":
    main()
