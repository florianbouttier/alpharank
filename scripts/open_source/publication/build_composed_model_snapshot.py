#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from alpharank.data.publishing.composed_snapshot import (
    build_composed_model_snapshot,
    validate_composed_model_snapshot,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Build one immutable SEC-only + hybrid-price model snapshot."
    )
    parser.add_argument("--price-package-dir", type=Path, required=True)
    parser.add_argument("--sec-package-dir", type=Path, required=True)
    parser.add_argument(
        "--history-root",
        type=Path,
        default=project_root / "data" / "production" / "history",
    )
    parser.add_argument(
        "--latest-manifest",
        type=Path,
        default=project_root / "data" / "production" / "latest.json",
    )
    parser.add_argument("--expected-through")
    args = parser.parse_args()

    result = build_composed_model_snapshot(
        price_package_dir=args.price_package_dir,
        sec_package_dir=args.sec_package_dir,
        history_root=args.history_root,
        latest_manifest_path=args.latest_manifest,
        expected_through=args.expected_through,
    )
    validation = validate_composed_model_snapshot(result.snapshot_dir)
    print(f"Composition id: {result.composition_id}")
    print(f"Snapshot: {result.snapshot_dir}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Validated files: {validation['file_count']}")


if __name__ == "__main__":
    main()
