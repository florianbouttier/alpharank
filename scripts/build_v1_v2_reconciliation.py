#!/usr/bin/env python3
"""Build the sealed v1-audited-biased versus v2-causal reconciliation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.replay import (
    build_v1_v2_reconciliation,
    validate_causal_v2_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--common-v2-dir", type=Path, required=True)
    parser.add_argument("--causal-snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    causal = validate_causal_v2_snapshot(args.causal_snapshot_dir)
    report = build_v1_v2_reconciliation(
        baseline_dir=args.baseline_dir,
        common_v2_dir=args.common_v2_dir,
        output_dir=args.output_dir,
        expected_composition_id=causal["composition_id"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
