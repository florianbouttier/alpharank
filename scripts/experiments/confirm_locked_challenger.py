#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpharank.multihorizon.confirmation import (
    cost_sensitivity,
    deflated_sharpe_statistics,
    holdings_and_concentration,
    meta_walk_forward_selection,
    paired_block_bootstrap,
    yearly_stability,
)
from alpharank.multihorizon.data import build_research_frame
from alpharank.multihorizon.legacy_ema import (
    legacy_winning_pairs,
    point_in_time_fold_features,
)
from alpharank.multihorizon.metrics import score_predictions
from alpharank.multihorizon.modeling import fit_booster
from alpharank.multihorizon.preprocessing import fit_fold_preprocessor
from alpharank.multihorizon.splits import horizon_walk_forward_windows
from alpharank.multihorizon.trading import (
    evaluate_trading_predictions,
    performance_statistics,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def _legacy_monthly(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .filter(pl.col("model") == "Combined_Frequency")
        .select(
            pl.col("year_month").cast(pl.Date).alias("holding_month"),
            pl.col("monthly_return").alias("legacy_return"),
        )
        .unique("holding_month")
        .sort("holding_month")
    )


def _performance_row(
    monthly: pl.DataFrame,
    *,
    strategy: str,
) -> dict:
    return {
        "strategy": strategy,
        "start_decision_month": monthly["decision_month"].min(),
        "end_decision_month": monthly["decision_month"].max(),
        "months": monthly.height,
        **{
            f"model_{key}": value
            for key, value in performance_statistics(
                monthly["net_return"].to_numpy()
            ).items()
        },
        **{
            f"benchmark_{key}": value
            for key, value in performance_statistics(
                monthly["benchmark_return"].to_numpy()
            ).items()
        },
        **{
            f"legacy_{key}": value
            for key, value in performance_statistics(
                monthly["legacy_return"].drop_nulls().to_numpy()
            ).items()
        },
    }


def _calibration_table(predictions: pl.DataFrame) -> pl.DataFrame:
    probability = predictions["calibrated_probability"].to_numpy()
    target = (
        predictions["future_excess_rank_6m"].to_numpy() >= 0.90
    ).astype(float)
    bins = np.clip(np.digitize(probability, np.linspace(0.0, 1.0, 11)) - 1, 0, 9)
    return (
        pl.DataFrame(
            {
                "calibration_bin": bins,
                "calibrated_probability": probability,
                "target": target,
            }
        )
        .group_by("calibration_bin")
        .agg(
            pl.len().alias("observations"),
            pl.col("calibrated_probability").mean().alias("mean_probability"),
            pl.col("target").mean().alias("observed_positive_rate"),
        )
        .sort("calibration_bin")
    )


def _fold_trading(
    predictions: pl.DataFrame,
    monthly: pl.DataFrame,
) -> pl.DataFrame:
    fold_by_month = predictions.select("decision_month", "fold").unique(
        "decision_month"
    )
    joined = monthly.join(fold_by_month, on="decision_month", how="left")
    rows: list[dict] = []
    for fold_frame in joined.partition_by("fold", maintain_order=True):
        rows.append(
            {
                "fold": int(fold_frame["fold"][0]),
                **_performance_row(fold_frame, strategy="locked_challenger"),
            }
        )
    return pl.DataFrame(rows).sort("fold")


def _evaluate_partial_holdout(
    specification: dict,
    *,
    output_dir: Path,
) -> dict:
    data = specification["data"]
    model = specification["model"]
    validation = specification["validation"]
    legacy_path = PROJECT_ROOT / data["legacy_detailed_returns"]
    pairs = legacy_winning_pairs(legacy_path)
    research = build_research_frame(
        data_dir=PROJECT_ROOT / data["input_snapshot"],
        legacy_detailed_returns_path=legacy_path,
        horizons=(1, int(model["horizon_months"])),
        start_month=validation["start_month"],
        excluded_tickers=("SII.US", "CBE.US", "TIE.US", "CPWR.US"),
        relative_ema_pairs=pairs,
    )
    horizon = int(model["horizon_months"])
    panel = research.frame.filter(
        pl.col(f"future_excess_return_{horizon}m").is_not_null()
        & pl.col("future_excess_return_1m").is_not_null()
    ).sort(["decision_month", "ticker"])
    months = panel["decision_month"].unique().sort().to_list()
    windows = horizon_walk_forward_windows(
        months,
        horizon=horizon,
        min_train_months=int(validation["min_train_months"]),
        validation_months=int(validation["validation_months"]),
        test_months=int(validation["test_months"]),
        step_months=int(validation["step_months"]),
    )
    last = windows[-1]
    train = panel.filter(pl.col("decision_month").is_in(last.train_months))
    validation_frame = panel.filter(
        pl.col("decision_month").is_in(last.validation_months)
    )
    last_exploratory_month = max(last.test_months)
    holdout = panel.filter(pl.col("decision_month") > last_exploratory_month)
    if holdout.is_empty():
        return {
            "available": False,
            "reason": "No mature decision month remains after the last exploratory block.",
            "last_exploratory_month": last_exploratory_month,
        }
    features, selected_pairs = point_in_time_fold_features(
        all_features=research.feature_columns,
        legacy_path=legacy_path,
        train_decision_cutoff=max(train["decision_month"]),
        include_non_relative_features=False,
    )
    preprocessor = fit_fold_preprocessor(
        train,
        features,
        max_missing_ratio=0.35,
    )
    _, X_train = preprocessor.transform(train)
    _, X_validation = preprocessor.transform(validation_frame)
    _, X_holdout = preprocessor.transform(holdout)
    fitted = fit_booster(
        method="classification",
        horizon=horizon,
        train_frame=train,
        validation_frame=validation_frame,
        X_train=X_train,
        X_validation=X_validation,
        features=preprocessor.features,
        positive_quantile=float(model["positive_quantile"]),
        seed=int(model["random_seed"]) + last.fold,
        num_boost_round=int(model["num_boost_round"]),
        params={},
    )
    predictions = holdout.select(
        "decision_month",
        "ticker",
        "legacy_selected",
        *[
            column
            for column in holdout.columns
            if column.startswith("future_") or column.startswith("benchmark_future_")
        ],
    ).with_columns(
        pl.Series("score", fitted.predict_raw_score(X_holdout)),
        pl.Series("calibrated_probability", fitted.predict(X_holdout)),
        pl.lit(last.fold + 1).alias("fold"),
        pl.lit("classification").alias("method"),
        pl.lit(horizon).alias("horizon"),
    )
    metrics, _ = score_predictions(
        predictions,
        method="classification",
        horizon=horizon,
        top_n_values=(5,),
    )
    trading_monthly, trading_summary = evaluate_trading_predictions(
        predictions,
        top_n_values=(5,),
        transaction_cost_bps=float(
            specification["portfolio"]["transaction_cost_bps_times_turnover"]
        ),
    )
    legacy = _legacy_monthly(PROJECT_ROOT / data["legacy_monthly_returns"])
    trading_monthly = trading_monthly.join(
        legacy,
        on="holding_month",
        how="left",
    )
    performance = _performance_row(
        trading_monthly.drop_nulls("legacy_return"),
        strategy="locked_partial_holdout",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(output_dir / "predictions.parquet")
    trading_monthly.write_csv(output_dir / "trading_monthly.csv")
    trading_summary.write_csv(output_dir / "trading_summary_raw.csv")
    pl.DataFrame([{"method": "classification", "horizon": horizon, **metrics}]).write_csv(
        output_dir / "model_metrics.csv"
    )
    pl.DataFrame([performance]).write_csv(output_dir / "performance.csv")
    manifest = {
        "available": True,
        "protocol": (
            "Reconstructed final locked fold without retuning, then extended its "
            "fixed model to mature months after the final exploratory test block."
        ),
        "training_start": min(train["decision_month"]),
        "training_cutoff": max(train["decision_month"]),
        "validation_start": min(validation_frame["decision_month"]),
        "validation_end": max(validation_frame["decision_month"]),
        "last_exploratory_month": last_exploratory_month,
        "holdout_start": min(holdout["decision_month"]),
        "holdout_end": max(holdout["decision_month"]),
        "holdout_months": holdout["decision_month"].n_unique(),
        "holdout_rows": holdout.height,
        "selected_pair_count": len(selected_pairs),
        "selected_pairs": selected_pairs,
        "feature_count": len(preprocessor.features),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    return {**manifest, **performance}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm the frozen exact-EMA challenger without changing its rule."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--champion-run", type=Path, required=True)
    parser.add_argument("--broad-run", type=Path, required=True)
    parser.add_argument("--ema-only-run", type=Path, required=True)
    parser.add_argument("--ema-plus-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--block-months", type=int, default=12)
    args = parser.parse_args()
    specification = json.loads(args.spec.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = PROJECT_ROOT / specification["data"]["legacy_detailed_returns"]
    monthly_path = PROJECT_ROOT / specification["data"]["legacy_monthly_returns"]
    detailed_hash = _sha256(detailed_path)
    monthly_hash = _sha256(monthly_path)
    if detailed_hash != specification["data"]["legacy_detailed_returns_sha256"]:
        raise ValueError("Locked detailed Legacy input hash does not match.")
    if monthly_hash != specification["data"]["legacy_monthly_returns_sha256"]:
        raise ValueError("Locked monthly Legacy input hash does not match.")

    combination = args.champion_run / "classification_h06"
    predictions = pl.read_parquet(combination / "predictions.parquet")
    monthly = pl.read_csv(combination / "trading_monthly.csv", try_parse_dates=True).filter(
        pl.col("top_n") == int(specification["portfolio"]["top_n"])
    )
    bootstrap = paired_block_bootstrap(
        monthly,
        comparator_columns={
            "S&P 500": "benchmark_return",
            "Legacy": "legacy_return",
        },
        samples=args.bootstrap_samples,
        block_months=args.block_months,
        seed=int(specification["model"]["random_seed"]),
    )
    bootstrap.write_csv(args.output_dir / "paired_block_bootstrap.csv")
    costs = cost_sensitivity(monthly, cost_bps_values=(0, 10, 25, 50, 100))
    costs.write_csv(args.output_dir / "cost_sensitivity.csv")
    yearly_stability(monthly).write_csv(args.output_dir / "yearly_stability.csv")
    _fold_trading(predictions, monthly).write_csv(
        args.output_dir / "fold_trading_stability.csv"
    )
    _calibration_table(predictions).write_csv(
        args.output_dir / "probability_calibration.csv"
    )
    dsr = deflated_sharpe_statistics(
        monthly["net_return"].to_numpy(),
        trials=162,
    )
    pl.DataFrame([dsr]).write_csv(args.output_dir / "deflated_sharpe.csv")

    general_path = (
        PROJECT_ROOT
        / specification["data"]["input_snapshot"]
        / "US_General.parquet"
    )
    holdings, ticker, sector, concentration = holdings_and_concentration(
        predictions,
        general_path=general_path,
        top_n=int(specification["portfolio"]["top_n"]),
    )
    holdings.write_parquet(args.output_dir / "holdings_monthly.parquet")
    ticker.write_csv(args.output_dir / "ticker_concentration.csv")
    sector.write_csv(args.output_dir / "sector_concentration.csv")
    concentration.write_csv(args.output_dir / "portfolio_concentration.csv")

    choices, meta_monthly, meta_summary = meta_walk_forward_selection(
        {
            "broad": args.broad_run,
            "ema_only": args.ema_only_run,
            "ema_plus": args.ema_plus_run,
        },
        horizons=(1, 3, 6, 12),
        methods=("classification", "regression", "ranking"),
        top_n_values=(5, 10, 20),
        lookback_months=36,
    )
    choices.write_csv(args.output_dir / "meta_selection_choices.csv")
    meta_monthly.write_csv(args.output_dir / "meta_selection_monthly.csv")
    meta_summary.write_csv(args.output_dir / "meta_selection_summary.csv")

    partial = _evaluate_partial_holdout(
        specification,
        output_dir=args.output_dir / "partial_holdout",
    )
    lock_audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "challenger_id": specification["challenger_id"],
        "locked_code_commit": specification["code_commit"],
        "current_repository_head": _git_head(),
        "spec_path": str(args.spec),
        "spec_sha256": _sha256(args.spec),
        "legacy_detailed_hash_verified": True,
        "legacy_monthly_hash_verified": True,
        "exploratory_performance": _performance_row(
            monthly,
            strategy="locked_challenger",
        ),
        "deflated_sharpe": dsr,
        "partial_holdout": partial,
        "disclosure": specification["selection_disclosure"],
    }
    (args.output_dir / "lock_audit.json").write_text(
        json.dumps(lock_audit, indent=2, default=str) + "\n"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
