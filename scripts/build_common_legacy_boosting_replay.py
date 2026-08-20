#!/usr/bin/env python3
"""Build a fail-closed Legacy/boosting/SPY replay on one common calendar."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpharank.portfolio.adapters.boosting import boosting_predictions_to_holdings
from alpharank.portfolio.artifacts import write_common_portfolio_artifacts
from alpharank.portfolio.comparison import reference_monthly_series
from alpharank.portfolio.lineage import (
    load_manifest,
    require_matching_data_contexts,
    require_matching_price_eligibility,
    require_matching_ticker_exclusions,
)
from alpharank.portfolio.maturity import split_completed_portfolio_months
from alpharank.portfolio.simulation import simulate_weighted_portfolio
from alpharank.portfolio.terminal_event_registry import load_terminal_event_registry
from alpharank.multihorizon.config import validate_latest_common_comparison_profile
from alpharank.data.terminal_eligibility import (
    TERMINAL_ENTRY_POLICY_ID,
    apply_terminal_entry_gate_to_decisions,
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _reference_series(source: pl.DataFrame, *, source_strategy: str, strategy: str) -> pl.DataFrame:
    return reference_monthly_series(
        source.filter(pl.col("strategy") == source_strategy),
        strategy=strategy,
        return_column="net_return",
    )


def _gate_boosting_terminal_entries(
    predictions: pl.DataFrame,
    *,
    entry_blocks: pl.DataFrame,
    prediction_partition: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if predictions.is_empty():
        return predictions.clone(), pl.DataFrame(
            schema={
                "prediction_partition": pl.String,
                "decision_month": pl.Date,
                "holding_month": pl.Date,
                "ticker": pl.String,
                "score": pl.Float64,
                "target_status_1m": pl.String,
                "terminal_event_id": pl.String,
                "blocked_from_holding_month": pl.Date,
                "entry_block_rule": pl.String,
            }
        )
    gate = apply_terminal_entry_gate_to_decisions(predictions, entry_blocks)
    eligible = gate.eligible.drop("holding_month")
    journal = gate.blocked.select(
        pl.lit(prediction_partition).alias("prediction_partition"),
        "decision_month",
        "holding_month",
        "ticker",
        *[
            column
            for column in ("score", "target_status_1m")
            if column in gate.blocked.columns
        ],
        "terminal_event_id",
        pl.col("_terminal_blocked_from_month").alias(
            "blocked_from_holding_month"
        ),
        "entry_block_rule",
    )
    return eligible, journal


def build_comparison(
    *,
    legacy_run_dir: Path,
    boosting_run_dir: Path,
    output_dir: Path,
    transaction_cost_bps: float,
) -> Path:
    legacy_manifest_path = legacy_run_dir / "data_input_manifest.json"
    boosting_manifest_path = boosting_run_dir / "manifest.json"
    predictions_path = boosting_run_dir / "classification_h06/predictions.parquet"
    legacy_monthly_path = legacy_run_dir / "legacy_common_monthly.parquet"
    legacy_holdings_path = legacy_run_dir / "legacy_common_holdings.parquet"
    for path in (
        legacy_manifest_path,
        boosting_manifest_path,
        predictions_path,
        legacy_monthly_path,
        legacy_holdings_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    boosting_manifest = load_manifest(boosting_manifest_path)
    comparison_profile = validate_latest_common_comparison_profile(
        boosting_manifest["config"]
    )
    if not comparison_profile["passed"]:
        raise ValueError(
            "Boosting run does not satisfy the public latest-common comparison "
            f"profile: {comparison_profile['mismatches']}"
        )
    required_keys = set(boosting_manifest["input_paths"])
    lineage = require_matching_data_contexts(
        legacy_manifest_path,
        boosting_manifest_path,
        required_keys=required_keys,
    )
    ticker_exclusion_check = require_matching_ticker_exclusions(
        legacy_manifest_path,
        boosting_manifest_path,
    )
    price_eligibility_check = require_matching_price_eligibility(
        legacy_manifest_path,
        boosting_manifest_path,
    )

    expected_snapshot = (legacy_run_dir / "input_snapshot").resolve()
    declared_snapshot = _declared_path(boosting_manifest["config"]["data_dir"]).resolve()
    if declared_snapshot != expected_snapshot:
        raise ValueError(
            "Legacy and boosting do not declare the same input snapshot: "
            f"{expected_snapshot} != {declared_snapshot}"
        )
    for key, filename in (
        ("legacy_detailed_returns", "legacy_detailed_returns_polars.parquet"),
        ("legacy_monthly_returns", "legacy_monthly_returns_polars.parquet"),
    ):
        source = legacy_run_dir / filename
        declared_hash = boosting_manifest[f"{key}_sha256"]
        if _hash(source) != declared_hash:
            raise ValueError(f"Boosting does not match {source.name} from the Legacy run.")

    all_predictions = pl.read_parquet(predictions_path)
    portfolio_maturity = split_completed_portfolio_months(all_predictions)
    terminal_registry = load_terminal_event_registry()
    completed_predictions_before_gate = portfolio_maturity.completed_predictions.height
    score_only_predictions_before_gate = portfolio_maturity.score_only_predictions.height
    predictions, completed_terminal_blocks = _gate_boosting_terminal_entries(
        portfolio_maturity.completed_predictions,
        entry_blocks=terminal_registry.terminal_entry_blocks(),
        prediction_partition="completed_replay",
    )
    score_only_predictions, score_only_terminal_blocks = (
        _gate_boosting_terminal_entries(
            portfolio_maturity.score_only_predictions,
            entry_blocks=terminal_registry.terminal_entry_blocks(),
            prediction_partition="score_only_live",
        )
    )
    terminal_entry_journal = pl.concat(
        [completed_terminal_blocks, score_only_terminal_blocks],
        how="diagonal_relaxed",
    ).sort(["holding_month", "prediction_partition", "ticker"])
    boosting_holdings = pl.concat(
        [
            boosting_predictions_to_holdings(
                predictions,
                strategy=f"Boosting Top {top_n}",
                top_n=top_n,
            )
            for top_n in (5, 10)
        ],
        how="diagonal_relaxed",
    )
    censored_selected = boosting_holdings.filter(
        pl.col("target_status_1m") == "approved_censored_last_observation"
    )
    if censored_selected.height:
        examples = censored_selected.select(
            "strategy", "decision_month", "holding_month", "ticker"
        ).head(10).to_dicts()
        raise ValueError(
            "Selected Boosting holdings still use censored zero-return targets "
            f"after the terminal entry gate: {examples}"
        )
    live_holdings_parts: list[pl.DataFrame] = []
    if not score_only_predictions.is_empty():
        for top_n in (5, 10):
            live_holdings_parts.append(
                boosting_predictions_to_holdings(
                    score_only_predictions,
                    strategy=f"Boosting Top {top_n}",
                    top_n=top_n,
                )
            )
    live_score_holdings = (
        pl.concat(live_holdings_parts, how="diagonal_relaxed")
        if live_holdings_parts
        else pl.DataFrame()
    )
    boosting_monthly = pl.concat(
        [
            simulate_weighted_portfolio(
                frame,
                transaction_cost_bps=transaction_cost_bps,
                causal_timing_policy="legacy_month_only",
            )
            for frame in boosting_holdings.partition_by("strategy", maintain_order=True)
        ],
        how="diagonal_relaxed",
    )
    if boosting_monthly.is_empty():
        raise ValueError("Boosting produced no realized portfolio month.")

    common_start = boosting_monthly["holding_month"].min()
    common_end = boosting_monthly["holding_month"].max()
    expected_months = (
        boosting_monthly.filter(pl.col("strategy") == "Boosting Top 5")
        .select("holding_month")
        .unique()
        .sort("holding_month")
    )
    legacy_monthly_source = pl.read_parquet(legacy_monthly_path).filter(
        pl.col("holding_month").is_between(common_start, common_end)
    )
    legacy_holdings = pl.read_parquet(legacy_holdings_path).filter(
        (pl.col("strategy") == "Combined_Frequency")
        & pl.col("holding_month").is_between(common_start, common_end)
    ).with_columns(pl.lit("Legacy").alias("strategy"))
    legacy_monthly = simulate_weighted_portfolio(
        legacy_holdings,
        transaction_cost_bps=transaction_cost_bps,
        causal_timing_policy="legacy_month_only",
    )
    references = pl.concat(
        [
            legacy_monthly,
            _reference_series(
                legacy_monthly_source,
                source_strategy="SPY total return",
                strategy="SPY total return",
            ),
        ],
        how="diagonal_relaxed",
    )
    for strategy in ("Legacy", "SPY total return"):
        observed = references.filter(pl.col("strategy") == strategy).select(
            "holding_month"
        )
        if observed.join(expected_months, on="holding_month", how="anti").height or (
            expected_months.join(observed, on="holding_month", how="anti").height
        ):
            raise ValueError(f"{strategy} does not cover the exact boosting calendar.")

    holdings = pl.concat([boosting_holdings, legacy_holdings], how="diagonal_relaxed")
    monthly = pl.concat([boosting_monthly, references], how="diagonal_relaxed")
    artifacts = write_common_portfolio_artifacts(
        output_dir=output_dir,
        holdings=holdings,
        monthly_returns=monthly,
        prefix="comparison_common",
        benchmark_metadata={
            "id": "spy_total_return_adjusted_close",
            "label": "SPY total return",
            "price_column": "adjusted_close",
            "includes_distributions": True,
        },
    )
    terminal_entry_journal_path = output_dir / "terminal_entry_journal.parquet"
    terminal_entry_journal_csv_path = output_dir / "terminal_entry_journal.csv"
    terminal_entry_journal.write_parquet(terminal_entry_journal_path)
    terminal_entry_journal.write_csv(terminal_entry_journal_csv_path)
    artifacts["terminal_entry_journal"] = terminal_entry_journal_path
    artifacts["terminal_entry_journal_csv"] = terminal_entry_journal_csv_path
    if not live_score_holdings.is_empty():
        live_score_holdings_path = output_dir / "boosting_live_score_holdings.parquet"
        live_score_holdings_csv_path = output_dir / "boosting_live_score_holdings.csv"
        live_score_holdings.write_parquet(live_score_holdings_path)
        live_score_holdings.write_csv(live_score_holdings_csv_path)
        artifacts["boosting_live_score_holdings"] = live_score_holdings_path
        artifacts["boosting_live_score_holdings_csv"] = live_score_holdings_csv_path

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "canonical_common_strategy_replay",
                "timing_contract": "decision_month=t; holding_month=t+1",
                "calendar": {
                    "start_holding_month": str(common_start),
                    "end_holding_month": str(common_end),
                    "months": expected_months.height,
                    "latest_month_status": "complete_realized_one_month_return",
                },
                "transaction_cost_policy": {
                    "strategies": ["Boosting Top 5", "Boosting Top 10", "Legacy"],
                    "bps_times_turnover": transaction_cost_bps,
                    "benchmark": "SPY total return has no simulated trading cost",
                },
                "comparison_profile": comparison_profile,
                "lineage_check": lineage,
                "ticker_exclusion_check": ticker_exclusion_check,
                "price_eligibility_check": price_eligibility_check,
                "sources": {
                    "legacy_run_manifest": {
                        "path": str(legacy_manifest_path.resolve()),
                        "sha256": _hash(legacy_manifest_path),
                    },
                    "boosting_manifest": {
                        "path": str(boosting_manifest_path.resolve()),
                        "sha256": _hash(boosting_manifest_path),
                    },
                    "boosting_predictions": {
                        "path": str(predictions_path.resolve()),
                        "sha256": _hash(predictions_path),
                    },
                },
                "model_calendar": {
                    "score_only_tail": boosting_manifest.get("protocol", {}).get(
                        "score_only_tail"
                    ),
                    "portfolio_maturity": portfolio_maturity.manifest,
                    "results": boosting_manifest.get("results"),
                },
                "terminal_entry_gate": {
                    "policy_id": TERMINAL_ENTRY_POLICY_ID,
                    "registry_id": terminal_registry.payload["registry_id"],
                    "registry_sha256": terminal_registry.sha256,
                    "registry_path": str(terminal_registry.path),
                    "completed_predictions_before_gate": completed_predictions_before_gate,
                    "completed_predictions_after_gate": predictions.height,
                    "completed_predictions_blocked": completed_terminal_blocks.height,
                    "score_only_predictions_before_gate": score_only_predictions_before_gate,
                    "score_only_predictions_after_gate": score_only_predictions.height,
                    "score_only_predictions_blocked": score_only_terminal_blocks.height,
                    "selected_censored_zero_return_rows": 0,
                },
                "artifacts": {
                    name: {"path": str(path.resolve()), "sha256": _hash(path)}
                    for name, path in artifacts.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-run-dir", type=Path, required=True)
    parser.add_argument("--boosting-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    args = parser.parse_args()
    print(
        build_comparison(
            legacy_run_dir=args.legacy_run_dir.resolve(),
            boosting_run_dir=args.boosting_run_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            transaction_cost_bps=args.transaction_cost_bps,
        )
    )


if __name__ == "__main__":
    main()
