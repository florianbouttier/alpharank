#!/usr/bin/env python3
"""Validate a sealed v1/v2 economic reconciliation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.reconciliation_v2 import validate_v1_v2_reconciliation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = validate_v1_v2_reconciliation(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
