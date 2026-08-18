#!/usr/bin/env python3
"""Validate a Legacy-v2 run against its sealed causal snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.causal_snapshot import validate_causal_v2_snapshot
from alpharank.legacy_v2 import validate_legacy_v2_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--causal-snapshot-dir", type=Path, required=True)
    args = parser.parse_args()
    causal = validate_causal_v2_snapshot(args.causal_snapshot_dir)
    result = validate_legacy_v2_replay(
        args.run_dir,
        expected_composition_id=causal["composition_id"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
