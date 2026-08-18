#!/usr/bin/env python3
"""Validate one causal common replay against its sealed snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.causal_snapshot import validate_causal_v2_snapshot
from alpharank.common_v2 import validate_common_v2_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--causal-snapshot-dir", type=Path, required=True)
    args = parser.parse_args()
    causal = validate_causal_v2_snapshot(args.causal_snapshot_dir)
    report = validate_common_v2_replay(
        args.output_dir,
        expected_composition_id=causal["composition_id"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
