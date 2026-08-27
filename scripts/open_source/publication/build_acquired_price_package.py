#!/usr/bin/env python3
"""Publish a completed acquisition run after review, without network access."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from alpharank.data.publishing.acquired_price_run import build_acquired_price_request
from alpharank.data.publishing.price_roll_forward import build_price_roll_forward_package


def main() -> None:
    args = _parse_args()
    request = build_acquired_price_request(
        run_dir=args.acquisition_run_dir,
        sec_package_dir=args.sec_package_dir,
        constituents_path=args.constituents_source,
        eodhd_seed_path=args.eodhd_seed,
        output_dir=args.output_dir,
        expected_through=args.expected_through,
        start_date=args.start_date,
        preserve_terminal_tickers=tuple(args.preserve_terminal_tickers),
        constituent_registry_path=args.constituent_registry,
        reviewed_move_registry_path=args.reviewed_extreme_price_moves,
        previous_lineage_path=args.previous_validated_lineage,
    )
    manifest = build_price_roll_forward_package(request)
    print(json.dumps({"output_dir": str(request.output_dir), "manifest": manifest}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-run-dir", type=Path, required=True)
    parser.add_argument("--sec-package-dir", type=Path, required=True)
    parser.add_argument(
        "--previous-validated-lineage",
        type=Path,
        help="Optional assertion of the baseline already bound by the acquisition run.",
    )
    parser.add_argument(
        "--constituents-source",
        type=Path,
        default=Path("data/SP500_Constituents.csv"),
    )
    parser.add_argument("--eodhd-seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-through", default=date.today().isoformat())
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument(
        "--preserve-terminal-tickers",
        nargs="*",
        default=(),
        help="Carry only tickers with a confirmed removal event in the registry.",
    )
    parser.add_argument(
        "--constituent-registry",
        type=Path,
        default=Path("configs/data_quality/sp500_constituent_changes_2026.json"),
    )
    parser.add_argument(
        "--reviewed-extreme-price-moves",
        type=Path,
        default=Path("configs/data_quality/reviewed_extreme_price_moves.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
