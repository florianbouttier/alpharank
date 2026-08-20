#!/usr/bin/env python3
"""Validate or explicitly regenerate the tracked Python dependency inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.quality.code_inventory import (
    build_code_inventory,
    validate_code_inventory,
    write_code_inventory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = PROJECT_ROOT / "docs/architecture/code_dependency_inventory_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    inventory_path = args.inventory.resolve()
    if args.write:
        write_code_inventory(inventory_path, build_code_inventory(root))
    report = validate_code_inventory(root, inventory_path)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
