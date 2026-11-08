#!/usr/bin/env python3
"""Build an audited Boosting Top 5 with at most one name per sector."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import polars as pl

from alpharank.data.terminal_eligibility import apply_terminal_entry_gate_to_decisions
from alpharank.portfolio.terminal_event_registry import load_terminal_event_registry
from alpharank.replay.sector_allocation import (
    SECTOR_CHAMPION_POLICY_ID,
    build_sector_champion_allocation,
)
from alpharank.replay.trend_eligibility import (
    build_causal_trend_eligibility_registry,
    filter_predictions_to_causal_trend_universe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boosting-run-dir", type=Path, required=True)
    parser.add_argument("--sector-history-release", type=Path, required=True)
    parser.add_argument("--market-prices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--causal-trend", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    predictions_path = args.boosting_run_dir / "classification_h06" / "predictions.parquet"
    sector_path = args.sector_history_release / "sector_history.parquet"
    predictions = pl.read_parquet(predictions_path)
    trend = (
        build_causal_trend_eligibility_registry(args.boosting_run_dir, predictions)
        if args.causal_trend
        else None
    )
    terminal_registry = load_terminal_event_registry()
    gate = apply_terminal_entry_gate_to_decisions(
        predictions,
        terminal_registry.terminal_entry_blocks(),
    )
    eligible_predictions = gate.eligible.drop("holding_month")
    strategy = "Boosting Top 5 | 1 per sector"
    if trend is not None:
        eligible_predictions = filter_predictions_to_causal_trend_universe(
            eligible_predictions,
            trend,
        )
        strategy = "Boosting Top 5 | Causal trend | 1 per sector"
    result = build_sector_champion_allocation(
        eligible_predictions,
        pl.read_parquet(sector_path),
        pl.read_parquet(args.market_prices),
        strategy=strategy,
    )
    args.output_dir.mkdir(parents=True)
    holdings_path = args.output_dir / "sector_champion_holdings.parquet"
    audit_path = args.output_dir / "sector_champion_audit.parquet"
    terminal_path = args.output_dir / "terminal_entry_blocks.parquet"
    result.holdings.write_parquet(holdings_path)
    result.audit.write_parquet(audit_path)
    gate.blocked.write_parquet(terminal_path)
    manifest = {
        "contract_version": 1,
        "policy_id": SECTOR_CHAMPION_POLICY_ID,
        "strategy": strategy,
        "causal_trend": args.causal_trend,
        "inputs": {
            "predictions_sha256": _sha256(predictions_path),
            "sector_history_sha256": _sha256(sector_path),
            "market_prices_sha256": _sha256(args.market_prices),
            "terminal_registry_sha256": terminal_registry.sha256,
        },
        "counts": {
            "candidate_rows": result.candidates.height,
            "holding_rows": result.holdings.height,
            "selected_months": result.audit.filter(pl.col("status") == "selected").height,
            "blocked_months": result.audit.filter(pl.col("status") != "selected").height,
            "terminal_entry_blocks": gate.blocked.height,
        },
        "files": {
            holdings_path.name: _sha256(holdings_path),
            audit_path.name: _sha256(audit_path),
            terminal_path.name: _sha256(terminal_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logging.info("output=%s", args.output_dir)
    logging.info("counts=%s", manifest["counts"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
