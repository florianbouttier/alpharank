#!/usr/bin/env python3
"""Build a canonical price package from a validated base plus one fresh vintage."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from alpharank.data.prices import resolve_previous_validated_price_lineage
from alpharank.data.publishing.price_package_output import PricePackageRequest
from alpharank.data.publishing.price_roll_forward import (
    build_price_roll_forward_package,
)


def main() -> None:
    args = _parse_args()
    base_dir = args.base_package_dir.resolve()
    base_manifest = _read_json(base_dir / "lineage" / "manifest.json")
    previous_source = None
    if args.previous_validated_lineage is not None:
        previous_lineage_path = args.previous_validated_lineage.resolve()
        previous_resolution = "explicit_cli_path"
        previous_composition_id = None
    else:
        previous_source = resolve_previous_validated_price_lineage(
            args.latest_composed_manifest.resolve()
        )
        previous_lineage_path = previous_source.lineage_path
        previous_resolution = "latest_composed_model_snapshot"
        previous_composition_id = previous_source.composition_id
    request = PricePackageRequest(
        run_id=str(base_manifest["run_id"]),
        source_refresh_contract=base_manifest["source_refresh_contract"],
        previous_lineage_path=previous_lineage_path,
        previous_resolution=previous_resolution,
        previous_composition_id=previous_composition_id,
        fresh_yahoo_path=args.fresh_yahoo_vintage.resolve(),
        benchmark_path=base_dir / "SP500Price.parquet",
        constituents_path=base_dir / "SP500_Constituents.csv",
        eodhd_seed_path=args.eodhd_seed.resolve(),
        output_dir=args.output_dir.resolve(),
        expected_through=args.expected_through,
        start_date=args.start_date,
        preserve_terminal_tickers=tuple(args.preserve_terminal_tickers),
        constituent_registry_path=args.constituent_registry.resolve(),
        reviewed_move_registry_path=args.reviewed_extreme_price_moves.resolve(),
        base_package_dir=base_dir,
        data_freshness=base_manifest["data_freshness"],
    )
    manifest = build_price_roll_forward_package(request)
    print(json.dumps({"output_dir": str(request.output_dir), "manifest": manifest}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-package-dir", type=Path, required=True)
    parser.add_argument(
        "--previous-validated-lineage",
        type=Path,
        help=(
            "Explicit prior lineage. If omitted, resolve it from the latest "
            "validated composed model snapshot."
        ),
    )
    parser.add_argument(
        "--latest-composed-manifest",
        type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "data"
            / "model_inputs"
            / "manifests"
            / "latest.json"
        ),
    )
    parser.add_argument("--fresh-yahoo-vintage", type=Path, required=True)
    parser.add_argument(
        "--reviewed-extreme-price-moves",
        type=Path,
        default=Path("configs/data_quality/reviewed_extreme_price_moves.json"),
    )
    parser.add_argument("--eodhd-seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-through", default=date.today().isoformat())
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument(
        "--preserve-terminal-tickers",
        nargs="*",
        default=(),
        help="Carry forward only tickers with a confirmed removal event in the registry.",
    )
    parser.add_argument(
        "--constituent-registry",
        type=Path,
        default=Path("configs/data_quality/sp500_constituent_changes_2026.json"),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
