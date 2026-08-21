#!/usr/bin/env python3
"""Build or validate legacy-to-MART reader migration evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.data.warehouse.reader_migration import (
    build_reader_migration_registry,
    validate_reader_migration_registry,
    write_reader_migration_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = (
    PROJECT_ROOT / "docs" / "architecture" / "data_location_inventory_v1.json"
)
DEFAULT_REGISTRY = (
    PROJECT_ROOT / "docs" / "architecture" / "data_reader_migration_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--observed-at", default="2026-08-20")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    inventory = json.loads(args.inventory.resolve().read_text(encoding="utf-8"))
    if not args.validate_only:
        write_reader_migration_registry(
            args.registry.resolve(),
            build_reader_migration_registry(
                root,
                inventory,
                observed_at=args.observed_at,
            ),
        )
    registry = json.loads(args.registry.resolve().read_text(encoding="utf-8"))
    report = validate_reader_migration_registry(root, inventory, registry)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
