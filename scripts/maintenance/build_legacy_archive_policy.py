#!/usr/bin/env python3
"""Build or validate the legacy data read-only observation policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.data.legacy_archive_policy import (
    build_legacy_archive_policy,
    validate_legacy_archive_policy,
    write_legacy_archive_policy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE = PROJECT_ROOT / "docs" / "architecture"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--catalog-summary",
        type=Path,
        default=ARCHITECTURE / "historical_data_migration_v1.json",
    )
    parser.add_argument(
        "--reader-registry",
        type=Path,
        default=ARCHITECTURE / "data_reader_migration_v1.json",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ARCHITECTURE / "legacy_data_archive_policy_v1.json",
    )
    parser.add_argument("--observation-started-at", default="2026-08-20")
    parser.add_argument("--minimum-observation-days", type=int, default=30)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    catalog = json.loads(args.catalog_summary.resolve().read_text(encoding="utf-8"))
    readers = json.loads(args.reader_registry.resolve().read_text(encoding="utf-8"))
    if not args.validate_only:
        write_legacy_archive_policy(
            args.policy.resolve(),
            build_legacy_archive_policy(
                args.root.resolve(),
                catalog,
                readers,
                observation_started_at=args.observation_started_at,
                minimum_observation_days=args.minimum_observation_days,
            ),
        )
    policy = json.loads(args.policy.resolve().read_text(encoding="utf-8"))
    report = validate_legacy_archive_policy(
        args.root.resolve(),
        catalog,
        readers,
        policy,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
