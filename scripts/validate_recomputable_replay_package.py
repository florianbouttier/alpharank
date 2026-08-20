#!/usr/bin/env python3
"""Validate and recompute a sealed common-portfolio replay package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.replay import validate_and_recompute_replay_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_and_recompute_replay_package(args.package_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
