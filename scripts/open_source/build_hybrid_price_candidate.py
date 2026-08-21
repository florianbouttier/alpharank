#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from alpharank.data.ingestion.prices import _load_latest_sp500_tickers
from alpharank.data.ingestion.storage import write_json
from alpharank.data.prices import (
    audit_price_candidate,
    compose_hybrid_price_history,
    load_eodhd_seed,
    validate_price_candidate,
)
from alpharank.data.prices.contracts import PRODUCTION_PRICE_GATE_POLICY

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build and audit a non-published hybrid price candidate from an "
            "immutable EODHD seed and one complete Yahoo active-universe vintage."
        )
    )
    parser.add_argument(
        "--eodhd-seed",
        type=Path,
        default=PROJECT_ROOT / "data" / "eodhd" / "output" / "US_Finalprice.parquet",
    )
    parser.add_argument("--yahoo-vintage", type=Path, required=True)
    parser.add_argument("--retained-open-lineage", type=Path, default=None)
    parser.add_argument(
        "--retained-open-runs-dir",
        type=Path,
        default=None,
        help="Immutable ingestion runs used to rebuild same-vintage daily returns.",
    )
    parser.add_argument(
        "--previous-prices",
        type=Path,
        default=PROJECT_ROOT / "data" / "open_source" / "output" / "US_Finalprice.parquet",
    )
    parser.add_argument(
        "--reference-data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
    )
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--expected-through", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-historical-price-revisions",
        action="store_true",
        help="Review mode only; this script never publishes the candidate.",
    )
    parser.add_argument(
        "--allow-historical-price-key-removals",
        action="store_true",
        help="Review mode only; preserve removed keys in the audit artifact.",
    )
    args = parser.parse_args()

    yahoo = pl.read_parquet(args.yahoo_vintage.resolve())
    _require_single_vintage(yahoo)
    previous = (
        pl.read_parquet(args.previous_prices.resolve())
        if args.previous_prices.exists()
        else None
    )
    active_tickers = _load_latest_sp500_tickers(args.reference_data_dir.resolve())
    seed = load_eodhd_seed(args.eodhd_seed.resolve(), start_date=args.start_date)
    retained = _load_retained_open_history(
        lineage_path=args.retained_open_lineage,
        runs_dir=args.retained_open_runs_dir,
        active_tickers=active_tickers,
    )
    yahoo_date = (
        pl.col("date").str.to_date(strict=False)
        if yahoo.schema.get("date") == pl.String
        else pl.col("date").cast(pl.Date, strict=False)
    )
    expected_through = args.expected_through or str(
        yahoo.select(yahoo_date.max()).item()
    )
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "outputs"
        / (
            "hybrid_price_candidate_"
            + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    lineage_dir = output_dir / "lineage"
    audit_dir = output_dir / "audit"
    lineage_dir.mkdir()
    audit_dir.mkdir()

    policy = replace(
        PRODUCTION_PRICE_GATE_POLICY,
        allow_historical_price_revisions=args.allow_historical_price_revisions,
        allow_historical_price_key_removals=(
            args.allow_historical_price_key_removals
        ),
    )
    hybrid = compose_hybrid_price_history(
        eodhd_seed=seed.frame,
        active_yahoo_vintage=yahoo,
        retained_open_history=retained,
        active_tickers=active_tickers,
        policy=policy,
    )
    gate = audit_price_candidate(
        previous_prices=previous,
        candidate_prices=hybrid.prices,
        candidate_lineage=hybrid.lineage,
        active_tickers=active_tickers,
        expected_eodhd_keys=seed.frame.select("ticker", "date"),
        expected_through=expected_through,
        policy=policy,
    )

    hybrid.prices.write_parquet(output_dir / "US_Finalprice.parquet")
    hybrid.lineage.write_parquet(
        lineage_dir / "prices_open_source_lineage.parquet"
    )
    gate.daily_return_revisions.write_parquet(
        audit_dir / "price_daily_return_revisions.parquet"
    )
    gate.transition_factor_findings.write_parquet(
        audit_dir / "price_transition_factor_findings.parquet"
    )
    gate.historical_key_removals.write_parquet(
        audit_dir / "price_historical_key_removals.parquet"
    )
    write_json(audit_dir / "price_composition.json", hybrid.composition_report)
    write_json(audit_dir / "price_revision_guard.json", gate.report)
    write_json(
        output_dir / "candidate_manifest.json",
        {
            "candidate_only": True,
            "published": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "eodhd_seed": seed.manifest(),
            "yahoo_vintage": str(args.yahoo_vintage.resolve()),
            "retained_open_lineage": (
                str(args.retained_open_lineage.resolve())
                if args.retained_open_lineage is not None
                else None
            ),
            "retained_open_runs_dir": (
                str(args.retained_open_runs_dir.resolve())
                if args.retained_open_runs_dir is not None
                else None
            ),
            "previous_prices": str(args.previous_prices.resolve()),
            "active_ticker_count": len(active_tickers),
            "start_date": args.start_date,
            "expected_through": expected_through,
            "composition": hybrid.composition_report,
            "gate": gate.report,
        },
    )
    validate_price_candidate(gate)
    print(f"Candidate: {output_dir}")
    print(f"Rows: {hybrid.prices.height}")
    print(f"Tickers: {hybrid.prices.select(pl.col('ticker').n_unique()).item()}")
    print(f"Gate passed: {gate.report['passed']}")


def _require_single_vintage(frame: pl.DataFrame) -> None:
    required = {"ticker", "date", "adjusted_close", "ingestion_run_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Yahoo vintage is missing columns: {sorted(missing)}")
    vintages = frame.select(pl.col("ingestion_run_id").drop_nulls().unique())
    if vintages.height != 1:
        raise RuntimeError(
            "Yahoo input must contain exactly one ingestion vintage; "
            f"found={vintages.height}."
        )


def _load_retained_open_history(
    *,
    lineage_path: Path | None,
    runs_dir: Path | None,
    active_tickers: tuple[str, ...],
) -> pl.DataFrame | None:
    active = [f"{ticker.upper().removesuffix('.US')}.US" for ticker in active_tickers]
    inputs: list[pl.LazyFrame] = []
    if lineage_path is not None:
        inputs.append(pl.scan_parquet(lineage_path.resolve()))
    if runs_dir is not None:
        run_paths = sorted(runs_dir.resolve().glob("*/raw/prices_yfinance.parquet"))
        inputs.extend(pl.scan_parquet(path) for path in run_paths)
    if not inputs:
        return None
    return pl.concat(inputs, how="diagonal_relaxed").filter(
        ~pl.col("ticker").cast(pl.String).str.to_uppercase().is_in(active)
    ).collect()


if __name__ == "__main__":
    main()
