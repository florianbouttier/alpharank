#!/usr/bin/env python3
"""Build or validate the versioned data-location and reader inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.quality.data_locations import (
    build_data_location_inventory,
    validate_data_location_inventory,
    write_data_location_inventory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = PROJECT_ROOT / "docs/architecture/data_location_inventory_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--observation-root",
        type=Path,
        help="Optional worktree whose data metadata is observed while code readers come from --root.",
    )
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--observed-at", default="2026-08-20")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    inventory_path = args.inventory.resolve()
    if args.write:
        write_data_location_inventory(
            inventory_path,
            build_data_location_inventory(
                root,
                observed_at=args.observed_at,
                observation_root=args.observation_root,
            ),
        )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    report = validate_data_location_inventory(root, inventory)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
