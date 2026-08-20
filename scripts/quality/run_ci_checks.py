#!/usr/bin/env python3
"""Run the same scoped AlphaRank quality checks locally and in CI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUFF_EXECUTABLE = str(Path(sys.executable).with_name("ruff"))
STATIC_CHECKS = (
    (
        "scripts/quality/check_ruff_baseline.py",
        "--ruff-executable",
        RUFF_EXECUTABLE,
    ),
    ("scripts/quality/check_error_handling.py",),
    ("scripts/quality/check_config_schemas.py",),
    ("scripts/quality/check_dependencies.py",),
    ("scripts/maintenance/build_code_inventory.py",),
    ("-m", "mypy"),
    ("scripts/validate_documentation.py",),
    ("scripts/validate_markdown_links.py", "."),
)
TARGETED_CI_TESTS = (
    (
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_config_schemas.py",
        "tests/test_code_inventory.py",
        "tests/test_dependency_sync.py",
        "tests/test_documentation_structure.py",
        "tests/test_error_handling_policy.py",
        "tests/test_future_mutation_invariance.py",
        "tests/test_methodology_ci.py",
        "tests/test_observability.py",
        "tests/test_recomputable_replay.py",
        "tests/test_ruff_baseline.py",
        "tests/test_script_path_independence.py",
        "tests/test_test_suite_classification.py",
    ),
)
LOCAL_GROUP_ORDER = ("static", "unit", "integration", "replay", "network", "production")
CHECK_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "ci": (*STATIC_CHECKS, *TARGETED_CI_TESTS),
    "static": STATIC_CHECKS,
    "unit": (("-m", "pytest", "-q", "-m", "unit", "-p", "no:cacheprovider"),),
    "integration": (("-m", "pytest", "-q", "-m", "integration", "-p", "no:cacheprovider"),),
    "replay": (("-m", "pytest", "-q", "-m", "replay", "-p", "no:cacheprovider"),),
    "network": (("-m", "pytest", "-q", "-m", "network", "-p", "no:cacheprovider"),),
    "production": (("-m", "pytest", "-q", "-m", "production", "-p", "no:cacheprovider"),),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=("all", *CHECK_GROUPS),
        default="all",
        help="Run one reproducible group or the complete ordered gate.",
    )
    args = parser.parse_args()

    groups = LOCAL_GROUP_ORDER if args.group == "all" else (args.group,)
    for group in groups:
        for command in CHECK_GROUPS[group]:
            completed = subprocess.run(
                [sys.executable, *command],
                cwd=PROJECT_ROOT,
                check=False,
            )
            if completed.returncode:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
