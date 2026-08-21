#!/usr/bin/env python3
"""Validate that a migration preserves the published portfolio prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from alpharank.governance import compare_economic_prefix


def _read_frame(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    raise ValueError(f"Unsupported economic artifact format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-holdings", type=Path, required=True)
    parser.add_argument("--candidate-holdings", type=Path, required=True)
    parser.add_argument("--reference-monthly", type=Path, required=True)
    parser.add_argument("--candidate-monthly", type=Path, required=True)
    parser.add_argument("--through-holding-month")
    parser.add_argument("--numeric-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--tolerance-justification",
        default=(
            "owner-approved floating serialization tolerance; "
            "structural decisions remain exact"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = compare_economic_prefix(
        reference_holdings=_read_frame(args.reference_holdings),
        candidate_holdings=_read_frame(args.candidate_holdings),
        reference_monthly=_read_frame(args.reference_monthly),
        candidate_monthly=_read_frame(args.candidate_monthly),
        through_holding_month=args.through_holding_month,
        numeric_tolerance=args.numeric_tolerance,
        tolerance_justification=args.tolerance_justification,
    )
    rendered = json.dumps(report, indent=2, default=str) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
