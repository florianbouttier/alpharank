#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from alpharank.data.warehouse.migration import migrate_validated_snapshot_to_warehouse


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Migrate the validated AlphaRank composition to RAW/STG/DEF/MART."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--latest-pointer",
        type=Path,
        default=project_root / "data" / "model_inputs" / "manifests" / "latest.json",
    )
    parser.add_argument(
        "--warehouse-root",
        type=Path,
        default=project_root / "data" / "warehouse",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Atomically move the production pointer after every migration check passes.",
    )
    args = parser.parse_args()
    result = migrate_validated_snapshot_to_warehouse(
        project_root=args.project_root,
        latest_pointer_path=args.latest_pointer,
        warehouse_root=args.warehouse_root,
        promote=args.promote,
    )
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(result).items()
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
