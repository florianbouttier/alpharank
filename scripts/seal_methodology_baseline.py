#!/usr/bin/env python3
"""Seal or validate an immutable methodology baseline package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.governance import (
    seal_baseline_package,
    validate_baseline_package,
)


def _source(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("source must use LABEL=PATH")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--baseline-id", default="v1-audited-biased")
    parser.add_argument("--source", action="append", type=_source, default=[])
    parser.add_argument("--approved-by", default="Florian Bouttier")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--source-snapshot-id")
    parser.add_argument("--known-limitation", action="append", default=[])
    args = parser.parse_args()

    if args.validate:
        report = validate_baseline_package(args.package_dir)
    else:
        if not args.implementation_commit:
            parser.error("--implementation-commit is required when sealing")
        manifest = seal_baseline_package(
            package_dir=args.package_dir,
            baseline_id=args.baseline_id,
            sources=dict(args.source),
            approved_by=args.approved_by,
            implementation_commit=args.implementation_commit,
            source_snapshot_id=args.source_snapshot_id,
            known_limitations=tuple(args.known_limitation),
        )
        report = {
            "baseline_id": manifest["baseline_id"],
            "payload_file_count": manifest["payload_file_count"],
            "payload_size_bytes": manifest["payload_size_bytes"],
            "payload_inventory_sha256": manifest["payload_inventory_sha256"],
            "passed": True,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
