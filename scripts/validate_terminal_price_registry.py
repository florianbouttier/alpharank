#!/usr/bin/env python3
"""Validate reviewed successor prices and optionally refetch SEC evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from alpharank.portfolio.terminal_price_registry import (
    DEFAULT_TERMINAL_PRICE_REGISTRY,
    load_terminal_price_registry,
    verify_terminal_price_source_hashes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_TERMINAL_PRICE_REGISTRY)
    parser.add_argument("--causal-snapshot-dir", type=Path, required=True)
    parser.add_argument("--verify-remote", action="store_true")
    args = parser.parse_args()

    registry = load_terminal_price_registry(args.registry)
    manifest = json.loads(
        (args.causal_snapshot_dir / "causal_v2_snapshot_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    price_path = args.causal_snapshot_dir / "input_snapshot" / "US_Finalprice.parquet"
    prices = registry.successor_prices(
        pl.read_parquet(price_path),
        snapshot_id=manifest["snapshot_id"],
        composition_id=manifest["source_snapshot"]["composition_id"],
        price_artifact_sha256=_sha256(price_path),
    )
    remote = (
        verify_terminal_price_source_hashes(registry)
        if args.verify_remote
        else []
    )
    print(
        json.dumps(
            {
                "passed": True,
                "registry_sha256": registry.sha256,
                "price_vintage_id": registry.price_vintage_id,
                "successor_price_rows": prices.height,
                "remote_sources_verified": len(remote),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
