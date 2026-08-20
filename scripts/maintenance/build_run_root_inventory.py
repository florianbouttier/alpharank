#!/usr/bin/env python3
"""Build or validate the immediate outputs directory inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.governance_contracts.run_organization import (
    build_run_root_inventory,
    validate_run_root_inventory,
    write_run_root_inventory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = (
    PROJECT_ROOT / "docs" / "architecture" / "run_root_inventory_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--observed-at", default="2026-08-20")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if not args.validate_only:
        write_run_root_inventory(
            args.inventory.resolve(),
            build_run_root_inventory(
                args.outputs_root.resolve(),
                observed_at=args.observed_at,
            ),
        )
    inventory = json.loads(args.inventory.resolve().read_text(encoding="utf-8"))
    report = validate_run_root_inventory(args.outputs_root.resolve(), inventory)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
