#!/usr/bin/env python3
"""Validate or explicitly regenerate strict schemas for maintained JSON configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.quality.config_schemas import (
    build_config_schema_registry,
    validate_config_schema_registry,
    write_config_schema_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs/data_contracts/config_schema_registry_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    registry_path = args.registry.resolve()
    if args.write:
        write_config_schema_registry(registry_path, build_config_schema_registry(root))
    report = validate_config_schema_registry(root, registry_path)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
