#!/usr/bin/env python3
"""Build the path-independent Pytest collection baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from alpharank.quality.test_catalog import tracked_test_paths
from alpharank.quality.test_collection import (
    build_collection_registry,
    collect_canonical_node_ids,
    write_collection_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "architecture" / "test_collection_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    root = args.root.resolve()
    registry = build_collection_registry(
        collect_canonical_node_ids(root, tracked_test_paths(root))
    )
    write_collection_registry(args.output.resolve(), registry)


if __name__ == "__main__":
    main()
