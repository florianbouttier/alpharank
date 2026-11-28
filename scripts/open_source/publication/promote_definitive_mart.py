#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from alpharank.data.warehouse.mart import validate_mart_model_input
from alpharank.data.warehouse.migration import (
    migrate_validated_snapshot_to_warehouse,
    promote_mart_pointer,
)


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
    parser.add_argument("--candidate-mart", type=Path)
    parser.add_argument("--replay-report", type=Path)
    args = parser.parse_args()
    if args.candidate_mart is not None:
        if not args.promote or args.replay_report is None:
            parser.error("--candidate-mart requires --promote and --replay-report")
        candidate = validate_mart_model_input(args.candidate_mart, warehouse_root=args.warehouse_root)
        promotion = promote_mart_pointer(
            pointer_path=args.latest_pointer,
            mart_dir=candidate.mart_dir,
            promotion_root=args.warehouse_root / "manifests" / "promotions",
            migration_id=f"live008_{candidate.composition_id[:12]}",
            generated_at=datetime.now(timezone.utc).isoformat(),
            replay_report_path=args.replay_report,
        )
        print(promotion)
        return 0
    if args.replay_report is not None:
        parser.error("--replay-report requires --candidate-mart")
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
