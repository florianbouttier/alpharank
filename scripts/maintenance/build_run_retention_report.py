#!/usr/bin/env python3
"""Measure exact output duplicates and write a reversible retention proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpharank.governance_contracts.run_retention import (
    build_run_retention_report,
    build_run_retention_summary,
    validate_run_retention_report,
    write_run_retention_report,
    write_run_retention_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "docs" / "architecture" / "run_retention_report_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--generated-at", default="2026-08-20")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    outputs_root = args.outputs_root.resolve()
    if args.validate_only and args.report is None:
        parser.error("--validate-only requires --report")
    if args.report is None:
        report = build_run_retention_report(
            outputs_root,
            generated_at=args.generated_at,
        )
        report_path = (
            PROJECT_ROOT
            / "data"
            / "warehouse"
            / "manifests"
            / "run_retention"
            / str(report["report_id"])
            / "manifest.json"
        )
        write_run_retention_report(report_path, report)
    else:
        report_path = args.report.resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
    validation = validate_run_retention_report(outputs_root, report)
    if not args.validate_only:
        write_run_retention_summary(
            args.summary.resolve(),
            build_run_retention_summary(report_path),
        )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
