#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpharank.data.ticker_integrity import load_ticker_exclusion_registry
from alpharank.multihorizon.confirmation import paired_block_bootstrap
from alpharank.multihorizon.data import build_research_frame
from alpharank.multihorizon.explain import compute_shap_sample, write_shap_outputs
from alpharank.multihorizon.legacy_ema import (
    legacy_winning_pairs,
    point_in_time_fold_features,
)
from alpharank.multihorizon.preprocessing import fit_fold_preprocessor
from alpharank.multihorizon.risk import (
    add_daily_forward_risk_targets,
    build_risk_weighted_backtest,
    fit_risk_booster,
    score_risk_predictions,
)
from alpharank.multihorizon.splits import horizon_walk_forward_windows
from alpharank.portfolio.performance import (
    legacy_report_statistics,
    performance_statistics,
)
from alpharank.portfolio.artifacts import write_common_portfolio_artifacts
from alpharank.portfolio.comparison import reference_monthly_series


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
    )


def _head_definitions(horizons: tuple[int, ...]) -> list[dict]:
    heads: list[dict] = []
    for horizon in horizons:
        heads.extend(
            [
                {
                    "head": "realized_volatility",
                    "horizon": horizon,
                    "task_type": "regression",
                    "target": f"future_realized_volatility_{horizon}m",
                    "prediction": f"predicted_realized_volatility_{horizon}m",
                    "probability": None,
                },
                {
                    "head": "daily_downside",
                    "horizon": horizon,
                    "task_type": "regression",
                    "target": f"future_daily_downside_{horizon}m",
                    "prediction": f"predicted_daily_downside_{horizon}m",
                    "probability": None,
                },
                {
                    "head": "high_volatility",
                    "horizon": horizon,
                    "task_type": "classification",
                    "target": f"future_realized_volatility_rank_{horizon}m",
                    "prediction": f"predicted_high_volatility_{horizon}m_score",
                    "probability": f"predicted_high_volatility_{horizon}m_probability",
                },
            ]
        )
    return heads


def _performance_rows(monthly: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    for strategy_frame in monthly.partition_by("strategy", maintain_order=True):
        rows.append(
            {
                "strategy": strategy_frame["strategy"][0],
                "start_decision_month": strategy_frame["decision_month"].min(),
                "end_decision_month": strategy_frame["decision_month"].max(),
                "months": strategy_frame.height,
                **{
                    f"model_{key}": value
                    for key, value in performance_statistics(
                        strategy_frame["net_return"].to_numpy()
                    ).items()
                },
                **{
                    f"benchmark_{key}": value
                    for key, value in performance_statistics(
                        strategy_frame["benchmark_return"].to_numpy()
                    ).items()
                },
                **{
                    f"legacy_{key}": value
                    for key, value in performance_statistics(
                        strategy_frame["legacy_return"].to_numpy()
                    ).items()
                },
                "average_turnover": float(strategy_frame["turnover"].mean()),
                "average_maximum_position_weight": float(
                    strategy_frame["maximum_position_weight"].mean()
                ),
                "maximum_sector_weight": float(
                    strategy_frame["maximum_sector_weight"].max()
                ),
                "average_sector_count": float(
                    strategy_frame["sector_count"].mean()
                ),
            }
        )
    return pl.DataFrame(rows).sort("strategy")


def _legacy_convention_performance_rows(monthly: pl.DataFrame) -> pl.DataFrame:
    """Compare every allocation and both references on one holding calendar."""

    rows: list[dict] = []
    for strategy_frame in monthly.partition_by("strategy", maintain_order=True):
        strategy = str(strategy_frame["strategy"][0])
        metrics = legacy_report_statistics(
            strategy_frame["net_return"].to_numpy(),
            holding_months=strategy_frame["holding_month"].to_list(),
        )
        rows.append(
            {
                "series": strategy,
                "role": "allocation",
                "start_holding_month": strategy_frame["holding_month"].min(),
                "end_holding_month": strategy_frame["holding_month"].max(),
                "months": strategy_frame.height,
                **metrics,
            }
        )

    reference_frame = monthly.filter(
        pl.col("strategy") == "alpha_top5_equal"
    ).sort("holding_month")
    for series, column in (
        ("Legacy", "legacy_return"),
        ("SPY total return", "benchmark_return"),
    ):
        metrics = legacy_report_statistics(
            reference_frame[column].to_numpy(),
            holding_months=reference_frame["holding_month"].to_list(),
        )
        rows.append(
            {
                "series": series,
                "role": "reference",
                "start_holding_month": reference_frame["holding_month"].min(),
                "end_holding_month": reference_frame["holding_month"].max(),
                "months": reference_frame.height,
                **metrics,
            }
        )
    return pl.DataFrame(rows).sort(["role", "series"])


def _reference_window_rows(
    monthly: pl.DataFrame,
    legacy_monthly_path: Path,
    sp500_price_path: Path,
) -> pl.DataFrame:
    """Expose full-history and ML-common Legacy/SPY performance separately."""

    legacy = (
        pl.read_parquet(legacy_monthly_path)
        .filter(pl.col("model") == "Combined_Frequency")
        .with_columns(
            pl.lit("Legacy").alias("series"),
            pl.col("year_month").cast(pl.Date).alias("holding_month"),
        )
        .select("series", "holding_month", "monthly_return")
    )
    sp500_prices = pl.read_parquet(sp500_price_path)
    spy_price_column = (
        "adjusted_close"
        if "adjusted_close" in sp500_prices.columns
        else "close"
    )
    spy = (
        sp500_prices.select(
            pl.col("date").cast(pl.Date, strict=False).alias("date"),
            pl.col(spy_price_column).cast(pl.Float64).alias("price"),
        )
        .drop_nulls()
        .sort("date")
        .with_columns(pl.col("date").dt.truncate("1mo").alias("holding_month"))
        .group_by("holding_month")
        .agg(pl.col("price").last().alias("price"))
        .sort("holding_month")
        .with_columns(
            pl.lit("SPY total return").alias("series"),
            pl.col("price").pct_change().alias("monthly_return"),
        )
        .select("series", "holding_month", "monthly_return")
        .drop_nulls("monthly_return")
    )
    references = pl.concat([legacy, spy])
    common_start = monthly["holding_month"].min()
    common_end = monthly["holding_month"].max()
    windows = (
        ("full_snapshot_common", None, None),
        ("ml_common", common_start, common_end),
        ("legacy_report_2015_2026", date(2015, 2, 1), date(2026, 4, 1)),
    )
    rows: list[dict] = []
    for window, requested_start, requested_end in windows:
        parts: dict[str, pl.DataFrame] = {}
        for series in ("Legacy", "SPY total return"):
            frame = references.filter(pl.col("series") == series)
            if requested_start is not None:
                frame = frame.filter(pl.col("holding_month") >= requested_start)
            if requested_end is not None:
                frame = frame.filter(pl.col("holding_month") <= requested_end)
            parts[series] = frame
        aligned_months = (
            parts["Legacy"]
            .select("holding_month")
            .join(
            parts["SPY total return"].select("holding_month"),
                on="holding_month",
                how="inner",
            )
        )
        for series in ("Legacy", "SPY total return"):
            frame = (
                parts[series]
                .join(aligned_months, on="holding_month", how="inner")
                .sort("holding_month")
            )
            metrics = legacy_report_statistics(
                frame["monthly_return"].to_numpy(),
                holding_months=frame["holding_month"].to_list(),
            )
            rows.append(
                {
                    "window": window,
                    "series": series,
                    "start_holding_month": frame["holding_month"].min(),
                    "end_holding_month": frame["holding_month"].max(),
                    "months": frame.height,
                    **metrics,
                }
            )
    return pl.DataFrame(rows).sort(["window", "series"])


def _cost_sensitivity(
    monthly: pl.DataFrame,
    *,
    cost_bps_values: tuple[float, ...],
) -> pl.DataFrame:
    rows: list[dict] = []
    for strategy_frame in monthly.partition_by("strategy", maintain_order=True):
        for cost_bps in cost_bps_values:
            net = (
                strategy_frame["gross_return"].to_numpy()
                - strategy_frame["turnover"].to_numpy()
                * cost_bps
                / 10_000.0
            )
            rows.append(
                {
                    "strategy": strategy_frame["strategy"][0],
                    "cost_bps": cost_bps,
                    **performance_statistics(net),
                }
            )
    return pl.DataFrame(rows).sort(["strategy", "cost_bps"])


def _bootstrap_rows(
    monthly: pl.DataFrame,
    *,
    primary_strategies: tuple[str, ...],
    samples: int,
    seed: int,
) -> pl.DataFrame:
    equal = monthly.filter(pl.col("strategy") == "alpha_top5_equal").select(
        "decision_month",
        pl.col("net_return").alias("equal_return"),
    )
    parts: list[pl.DataFrame] = []
    for strategy in primary_strategies:
        frame = (
            monthly.filter(pl.col("strategy") == strategy)
            .join(equal, on="decision_month", how="inner")
            .sort("decision_month")
        )
        comparators = {
            "SPY total return": "benchmark_return",
            "Legacy": "legacy_return",
        }
        if strategy != "alpha_top5_equal":
            comparators["alpha_top5_equal"] = "equal_return"
        parts.append(
            paired_block_bootstrap(
                frame,
                comparator_columns=comparators,
                samples=samples,
                block_months=12,
                seed=seed,
            ).with_columns(pl.lit(strategy).alias("strategy"))
        )
    return pl.concat(parts, how="diagonal_relaxed")


def _acceptance_gates(
    performance: pl.DataFrame,
    costs: pl.DataFrame,
) -> pl.DataFrame:
    baseline = performance.filter(
        pl.col("strategy") == "alpha_top5_equal"
    ).row(0, named=True)
    baseline_50 = costs.filter(
        (pl.col("strategy") == "alpha_top5_equal")
        & (pl.col("cost_bps") == 50.0)
    ).row(0, named=True)
    rows: list[dict] = []
    for strategy in (
        "alpha_top5_inverse_vol_h3",
        "alpha_top5_inverse_vol_h3_sector2",
    ):
        row = performance.filter(pl.col("strategy") == strategy).row(
            0,
            named=True,
        )
        row_50 = costs.filter(
            (pl.col("strategy") == strategy)
            & (pl.col("cost_bps") == 50.0)
        ).row(0, named=True)
        checks = {
            "sharpe_higher": row["model_sharpe"] > baseline["model_sharpe"],
            "drawdown_improves_5pp": (
                row["model_max_drawdown"]
                - baseline["model_max_drawdown"]
                >= 0.05
            ),
            "cagr_loss_within_3pp": (
                row["model_cagr"] >= baseline["model_cagr"] - 0.03
            ),
            "sector_weight_at_most_40pct": (
                row["maximum_sector_weight"] <= 0.40 + 1e-10
            ),
            "sharpe_higher_at_50bps": (
                row_50["sharpe"] > baseline_50["sharpe"]
            ),
        }
        rows.append(
            {
                "strategy": strategy,
                **checks,
                "all_gates_pass": all(checks.values()),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit exact-EMA boosting risk heads and test risk-aware sizing."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--shap-sample-per-fold", type=int, default=40)
    args = parser.parse_args()

    args.spec = args.spec.resolve()
    specification = json.loads(args.spec.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / specification["data"]["input_snapshot"]
    legacy_detailed_path = (
        PROJECT_ROOT / specification["data"]["legacy_detailed_returns"]
    )
    legacy_monthly_path = (
        PROJECT_ROOT / specification["data"]["legacy_monthly_returns"]
    )
    alpha_run = PROJECT_ROOT / specification["alpha"]["prediction_run"]
    horizons = tuple(
        int(value)
        for value in specification["risk_heads"]["horizons_months"]
    )
    pairs = legacy_winning_pairs(legacy_detailed_path)
    research = build_research_frame(
        data_dir=data_dir,
        legacy_detailed_returns_path=legacy_detailed_path,
        horizons=(1, int(specification["alpha"]["horizon_months"])),
        start_month=specification["validation"]["start_month"],
        excluded_tickers=tuple(
            specification["data"].get(
                "excluded_tickers",
                load_ticker_exclusion_registry().excluded_tickers,
            )
        ),
        relative_ema_pairs=pairs,
        minimum_monthly_price_observations=int(
            specification["data"]
            .get("price_integrity", {})
            .get("minimum_monthly_price_observations", 1)
        ),
        minimum_monthly_median_dollar_volume=float(
            specification["data"]
            .get("price_integrity", {})
            .get("minimum_monthly_median_dollar_volume", 0.0)
        ),
        maximum_monthly_ohlc_violation_rate=float(
            specification["data"]
            .get("price_integrity", {})
            .get("maximum_monthly_ohlc_violation_rate", 1.0)
        ),
    )
    final_price = pl.read_parquet(research.input_paths["final_price"])
    frame = add_daily_forward_risk_targets(
        research.frame,
        final_price=final_price,
        horizons=horizons,
    )
    del final_price

    alpha_predictions = pl.read_parquet(
        alpha_run / "classification_h06" / "predictions.parquet"
    )
    alpha_horizon = int(specification["alpha"]["horizon_months"])
    panel = frame.filter(
        pl.col(f"future_excess_return_{alpha_horizon}m").is_not_null()
        & pl.col("future_return_1m").is_not_null()
    ).sort(["decision_month", "ticker"])
    months = panel["decision_month"].unique().sort().to_list()
    windows = horizon_walk_forward_windows(
        months,
        horizon=alpha_horizon,
        min_train_months=int(
            specification["validation"]["min_train_months"]
        ),
        validation_months=int(
            specification["validation"]["validation_months"]
        ),
        test_months=int(specification["validation"]["test_months"]),
        step_months=int(specification["validation"]["step_months"]),
        include_partial_test_window=bool(
            specification["validation"][
                "include_partial_final_test_window"
            ]
        ),
    )
    expected_test_months = {
        month for window in windows for month in window.test_months
    }
    observed_test_months = set(alpha_predictions["decision_month"].unique())
    if expected_test_months != observed_test_months:
        raise ValueError("Risk windows do not match the frozen alpha run.")

    heads = _head_definitions(horizons)
    prediction_parts: dict[str, list[pl.DataFrame]] = {
        item["prediction"]: [] for item in heads
    }
    fold_metric_rows: list[dict] = []
    feature_rows: list[dict] = []
    shap_parts: dict[str, list[pl.DataFrame]] = {
        item["prediction"]: []
        for item in heads
        if item["horizon"]
        == int(
            specification["risk_heads"][
                "primary_allocation_horizon_months"
            ]
        )
    }
    for window in windows:
        train_base = panel.filter(
            pl.col("decision_month").is_in(window.train_months)
        )
        validation_base = panel.filter(
            pl.col("decision_month").is_in(window.validation_months)
        )
        fold_alpha = alpha_predictions.filter(pl.col("fold") == window.fold)
        test_base = panel.join(
            fold_alpha.select("decision_month", "ticker"),
            on=["decision_month", "ticker"],
            how="inner",
        )
        train_cutoff = max(train_base["decision_month"])
        fold_features, fold_pairs = point_in_time_fold_features(
            all_features=research.feature_columns,
            legacy_path=legacy_detailed_path,
            train_decision_cutoff=train_cutoff,
            include_non_relative_features=False,
        )
        for item in heads:
            target = item["target"]
            train = train_base.filter(pl.col(target).is_not_null())
            validation = validation_base.filter(pl.col(target).is_not_null())
            test = test_base
            preprocessor = fit_fold_preprocessor(
                train,
                fold_features,
                max_missing_ratio=0.35,
            )
            _, X_train = preprocessor.transform(train)
            _, X_validation = preprocessor.transform(validation)
            _, X_test = preprocessor.transform(test)
            fitted = fit_risk_booster(
                task_type=item["task_type"],
                target_column=target,
                train_frame=train,
                validation_frame=validation,
                X_train=X_train,
                X_validation=X_validation,
                features=preprocessor.features,
                seed=int(specification["alpha"]["random_seed"])
                + window.fold,
                num_boost_round=int(
                    specification["risk_heads"]["num_boost_round"]
                ),
            )
            raw = fitted.predict_raw_score(X_test)
            output = test.select(
                "decision_month",
                "ticker",
                target,
            ).with_columns(
                pl.Series(item["prediction"], raw),
                pl.lit(window.fold).alias("fold"),
            )
            if item["probability"]:
                output = output.with_columns(
                    pl.Series(item["probability"], fitted.predict(X_test))
                )
            prediction_parts[item["prediction"]].append(output)
            metric_source = output.filter(pl.col(target).is_not_null())
            fold_metric_rows.append(
                {
                    "fold": window.fold,
                    **{
                        key: item[key]
                        for key in ("head", "horizon", "task_type", "target")
                    },
                    **score_risk_predictions(
                        metric_source,
                        target_column=target,
                        prediction_column=item["prediction"],
                        task_type=item["task_type"],
                        probability_column=item["probability"],
                    ),
                }
            )
            feature_rows.append(
                {
                    "fold": window.fold,
                    "head": item["head"],
                    "horizon": item["horizon"],
                    "train_start": min(train["decision_month"]),
                    "train_cutoff": max(train["decision_month"]),
                    "validation_start": min(validation["decision_month"]),
                    "validation_end": max(validation["decision_month"]),
                    "test_start": min(test["decision_month"]),
                    "test_end": max(test["decision_month"]),
                    "winner_pair_count": len(fold_pairs),
                    "feature_count": len(preprocessor.features),
                    "features": json.dumps(preprocessor.features),
                }
            )
            if item["prediction"] in shap_parts:
                shap_parts[item["prediction"]].append(
                    compute_shap_sample(
                        fitted=fitted,
                        X=X_test,
                        source=test,
                        fold=window.fold,
                        method=f"risk_{item['head']}",
                        horizon=item["horizon"],
                        sample_size=args.shap_sample_per_fold,
                        seed=int(specification["alpha"]["random_seed"])
                        + window.fold,
                    )
                )

    base_columns = [
        "decision_month",
        "ticker",
        "fold",
        "score",
        "calibrated_probability",
        "legacy_selected",
        "future_return_1m",
        "benchmark_future_return_1m",
        "future_excess_return_1m",
        "future_excess_rank_6m",
    ]
    combined = alpha_predictions.select(base_columns)
    overall_metric_rows: list[dict] = []
    for item in heads:
        head_predictions = pl.concat(
            prediction_parts[item["prediction"]]
        ).sort(["decision_month", "ticker"])
        metric_source = head_predictions.filter(
            pl.col(item["target"]).is_not_null()
        )
        overall_metric_rows.append(
            {
                **{
                    key: item[key]
                    for key in ("head", "horizon", "task_type", "target")
                },
                "test_months": metric_source[
                    "decision_month"
                ].n_unique(),
                "test_rows": metric_source.height,
                **score_risk_predictions(
                    metric_source,
                    target_column=item["target"],
                    prediction_column=item["prediction"],
                    task_type=item["task_type"],
                    probability_column=item["probability"],
                ),
            }
        )
        keep = [
            "decision_month",
            "ticker",
            item["target"],
            item["prediction"],
        ]
        if item["probability"]:
            keep.append(item["probability"])
        combined = combined.join(
            head_predictions.select(keep),
            on=["decision_month", "ticker"],
            how="left",
        )

    combined.write_parquet(args.output_dir / "risk_predictions.parquet")
    pl.DataFrame(overall_metric_rows).sort(
        ["head", "horizon"]
    ).write_csv(args.output_dir / "risk_model_metrics.csv")
    pl.DataFrame(fold_metric_rows).sort(
        ["head", "horizon", "fold"]
    ).write_csv(args.output_dir / "risk_fold_metrics.csv")
    pl.DataFrame(feature_rows).sort(
        ["head", "horizon", "fold"]
    ).write_csv(args.output_dir / "risk_feature_manifest.csv")
    for prediction_column, parts in shap_parts.items():
        write_shap_outputs(
            pl.concat(parts, how="diagonal_relaxed"),
            args.output_dir / "shap" / prediction_column,
            top_features=20,
        )

    general = pl.read_parquet(data_dir / "US_General.parquet")
    transaction_cost_bps = float(
        specification["trading"]["transaction_cost_bps_times_turnover"]
    )
    strategies: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    strategies.append(
        build_risk_weighted_backtest(
            combined,
            general=general,
            strategy="alpha_top5_equal",
            transaction_cost_bps=transaction_cost_bps,
        )
    )
    for horizon in horizons:
        strategies.append(
            build_risk_weighted_backtest(
                combined,
                general=general,
                strategy=f"alpha_top5_inverse_vol_h{horizon}",
                risk_column=f"predicted_realized_volatility_{horizon}m",
                transaction_cost_bps=transaction_cost_bps,
            )
        )
        strategies.append(
            build_risk_weighted_backtest(
                combined,
                general=general,
                strategy=f"alpha_top5_inverse_downside_h{horizon}",
                risk_column=f"predicted_daily_downside_{horizon}m",
                transaction_cost_bps=transaction_cost_bps,
            )
        )
    primary_horizon = int(
        specification["risk_heads"]["primary_allocation_horizon_months"]
    )
    strategies.append(
        build_risk_weighted_backtest(
            combined,
            general=general,
            strategy=(
                f"alpha_top5_inverse_vol_h{primary_horizon}_sector2"
            ),
            risk_column=(
                f"predicted_realized_volatility_{primary_horizon}m"
            ),
            maximum_names_per_sector=2,
            maximum_sector_weight=0.40,
            transaction_cost_bps=transaction_cost_bps,
        )
    )
    legacy = _legacy_monthly(legacy_monthly_path)
    monthly = pl.concat([item[0] for item in strategies]).join(
        legacy,
        on="holding_month",
        how="inner",
    )
    holdings = pl.concat(
        [item[1] for item in strategies],
        how="diagonal_relaxed",
    )
    performance = _performance_rows(monthly)
    legacy_convention_performance = _legacy_convention_performance_rows(monthly)
    reference_windows = _reference_window_rows(
        monthly,
        legacy_monthly_path,
        data_dir / "SP500Price.parquet",
    )
    costs = _cost_sensitivity(
        monthly,
        cost_bps_values=tuple(
            float(value)
            for value in specification["trading"]["cost_sensitivity_bps"]
        ),
    )
    primary_strategies = (
        "alpha_top5_equal",
        f"alpha_top5_inverse_vol_h{primary_horizon}",
        f"alpha_top5_inverse_vol_h{primary_horizon}_sector2",
    )
    bootstrap = _bootstrap_rows(
        monthly,
        primary_strategies=primary_strategies,
        samples=args.bootstrap_samples,
        seed=int(specification["alpha"]["random_seed"]),
    )
    gates = _acceptance_gates(performance, costs)
    monthly.write_csv(args.output_dir / "allocation_monthly.csv")
    holdings.write_parquet(args.output_dir / "allocation_holdings.parquet")
    common_monthly = monthly.with_columns(
        (pl.col("net_return") - pl.col("benchmark_return")).alias("active_return"),
        (
            (1.0 + pl.col("net_return")) / (1.0 + pl.col("benchmark_return")) - 1.0
        ).alias("relative_return"),
    )
    reference_source = common_monthly.filter(pl.col("strategy") == "alpha_top5_equal")
    common_monthly = pl.concat(
        [
            common_monthly,
            reference_monthly_series(
                reference_source,
                strategy="Legacy",
                return_column="legacy_return",
            ),
            reference_monthly_series(
                reference_source,
                strategy="SPY total return",
                return_column="benchmark_return",
            ),
        ],
        how="diagonal_relaxed",
    )
    write_common_portfolio_artifacts(
        output_dir=args.output_dir,
        holdings=holdings.select(
            "strategy",
            "decision_month",
            "holding_month",
            "ticker",
            "target_weight",
            "realized_return",
            "benchmark_return",
            "sector",
            "selection_rank",
            "score",
        ),
        monthly_returns=common_monthly.select(
            "strategy",
            "decision_month",
            "holding_month",
            "gross_return",
            "turnover",
            "transaction_cost",
            "net_return",
            "benchmark_return",
            "active_return",
            "relative_return",
            "n_positions",
            "maximum_position_weight",
            "maximum_sector_weight",
            "sector_count",
        ),
    )
    performance.write_csv(args.output_dir / "allocation_performance.csv")
    legacy_convention_performance.write_csv(
        args.output_dir / "allocation_performance_legacy_convention.csv"
    )
    reference_windows.write_csv(
        args.output_dir / "reference_performance_windows.csv"
    )
    costs.write_csv(args.output_dir / "allocation_cost_sensitivity.csv")
    bootstrap.write_csv(args.output_dir / "allocation_paired_bootstrap.csv")
    gates.write_csv(args.output_dir / "allocation_acceptance_gates.csv")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_id": specification["research_id"],
        "spec_path": str(args.spec),
        "spec_sha256": _sha256(args.spec),
        "repository_head": _git_head(),
        "code_context": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in (
                PROJECT_ROOT / "src/alpharank/multihorizon/risk.py",
                PROJECT_ROOT
                / "scripts/experiments/run_legacy_ema_risk_heads.py",
                args.spec,
            )
        },
        "input_paths": {
            key: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for key, path in research.input_paths.items()
        },
        "legacy_detailed": {
            "path": str(legacy_detailed_path),
            "sha256": _sha256(legacy_detailed_path),
        },
        "legacy_monthly": {
            "path": str(legacy_monthly_path),
            "sha256": _sha256(legacy_monthly_path),
        },
        "alpha_run": str(alpha_run),
        "test_start": combined["decision_month"].min(),
        "test_end": combined["decision_month"].max(),
        "test_months": combined["decision_month"].n_unique(),
        "test_rows": combined.height,
        "outer_folds": combined["fold"].n_unique(),
        "risk_targets": {
            "realized_volatility": (
                "annualized sample standard deviation of strictly future "
                "daily returns; every future month requires at least 10 "
                "daily observations"
            ),
            "daily_downside": (
                "annualized square root of the mean squared negative strictly "
                "future daily return"
            ),
            "high_volatility": (
                "cross-sectional top 20% future realized-volatility label"
            ),
        },
        "selection_disclosure": specification["risk_heads"][
            "selection_disclosure"
        ],
        "all_primary_acceptance_gates_pass": bool(
            gates["all_gates_pass"].all()
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
