#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.data.sources.sec_raw_versions import (
    build_sec_raw_version_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild a raw SEC candidate while preserving filing versions."
    )
    parser.add_argument("--retained-raw-dir", type=Path, required=True)
    parser.add_argument("--run-raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_sec_raw_version_candidate(
        retained_raw_dir=args.retained_raw_dir,
        run_raw_dir=args.run_raw_dir,
        output_dir=args.output_dir,
    )
    manifest_path = args.output_dir / "version_rebuild_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
