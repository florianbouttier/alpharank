#!/usr/bin/env python3
"""Validate or regenerate the tracked Python directory-size inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.quality.python_directories import (
    build_python_directory_inventory,
    load_python_directory_policy,
    validate_python_directory_inventory,
    write_python_directory_inventory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = PROJECT_ROOT / "configs/quality/python_directory_policy_v1.json"
DEFAULT_INVENTORY = PROJECT_ROOT / "docs/architecture/python_directory_inventory_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--enforce-limit", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    policy_path = args.policy.resolve()
    inventory_path = args.inventory.resolve()
    if args.write:
        policy = load_python_directory_policy(policy_path)
        write_python_directory_inventory(
            inventory_path,
            build_python_directory_inventory(root, policy),
        )
    report = validate_python_directory_inventory(
        root,
        policy_path,
        inventory_path,
        enforce_limit=args.enforce_limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
