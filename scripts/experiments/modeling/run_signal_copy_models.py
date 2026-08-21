from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import polars as pl

from alpharank.backtest.time_folds import CombinatorialPurgedGroupTimeSeriesSplit, filter_by_months, walk_forward_windows
from alpharank.backtest.tuning import safe_auc
from alpharank.utils.xgboost_runtime import load_xgboost


DEFAULT_SOURCE_RUN = Path("outputs/xgboost_timefold_backtest_20260612_175250")
DEFAULT_LEGACY_PATH = Path("outputs/2026-06-07/legacy_detailed_returns_polars.parquet")

IDENTITY_COLS = [
    "ticker",
    "year_month",
    "decision_month",
    "holding_month",
    "decision_asof_date",
    "holding_asof_date",
    "benchmark_holding_asof_date",
    "holding_period_complete",
    "monthly_return",
    "future_return",
    "benchmark_future_return",
    "future_excess_return",
    "future_relative_return",
]


@dataclass(frozen=True)
class SignalCopyConfig:
    source_run: Path = DEFAULT_SOURCE_RUN
    legacy_path: Path = DEFAULT_LEGACY_PATH
    output_dir: Path = Path("outputs")
    max_windows: int = 12
    min_train_months: int = 24
    val_months: int = 12
    test_months: int = 1
    step_months: int = 1
    top_n: int = 10
    gate_n: int = 50
    threshold: float = 0.05
    inner_groups: int = 5
    seed: int = 42


def _binary_target(frame: pl.DataFrame, threshold: float) -> np.ndarray:
    return (frame.get_column("future_excess_return").to_numpy() > float(threshold)).astype(np.int8)


def _matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    return frame.select(list(features)).to_numpy().astype(np.float32, copy=False)


def _logit(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-5, 1.0 - 1e-5)
    return np.log(prob / (1.0 - prob))


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = float(np.nanstd(arr))
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - float(np.nanmean(arr))) / std


def _rank_pct(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float))
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1, dtype=float)
    return ranks / max(1, order.size)


def _month_rank_target(frame: pl.DataFrame, value_col: str) -> np.ndarray:
    ranked = frame.with_columns(pl.col(value_col).rank(method="average").over("year_month").alias("__rank"))
    counts = ranked.group_by("year_month").agg(pl.len().alias("__n"))
    ranked = ranked.join(counts, on="year_month", how="left")
    return (ranked.get_column("__rank").to_numpy() / ranked.get_column("__n").to_numpy()).astype(np.float32)


def _train_xgb(
    xgb,
    X: np.ndarray,
    y: np.ndarray,
    *,
    params: dict,
    rounds: int,
    sample_weight: np.ndarray | None = None,
    base_margin: np.ndarray | None = None,
    group: Sequence[int] | None = None,
):
    dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
    if base_margin is not None:
        dtrain.set_base_margin(base_margin)
    if group is not None:
        dtrain.set_group(list(group))
    return xgb.train(params=params, dtrain=dtrain, num_boost_round=int(rounds), verbose_eval=False)


def _predict_xgb(xgb, model, X: np.ndarray, *, base_margin: np.ndarray | None = None) -> np.ndarray:
    dtest = xgb.DMatrix(X)
    if base_margin is not None:
        dtest.set_base_margin(base_margin)
    return np.asarray(model.predict(dtest), dtype=float)


def _classifier_params(seed: int, *, monotone_constraints: str | None = None) -> dict:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.025,
        "max_depth": 3,
        "subsample": 0.75,
        "colsample_bytree": 0.85,
        "min_child_weight": 5.0,
        "gamma": 2.0,
        "alpha": 1.5,
        "lambda": 3.0,
        "seed": seed,
        "verbosity": 0,
        "nthread": -1,
    }
    if monotone_constraints is not None:
        params["monotone_constraints"] = monotone_constraints
    return params


def _residual_params(seed: int) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.02,
        "max_depth": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 8.0,
        "gamma": 3.0,
        "alpha": 2.0,
        "lambda": 5.0,
        "seed": seed,
        "verbosity": 0,
        "nthread": -1,
    }


def _regression_params(seed: int) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,
        "max_depth": 3,
        "subsample": 0.75,
        "colsample_bytree": 0.85,
        "min_child_weight": 5.0,
        "gamma": 1.0,
        "alpha": 1.0,
        "lambda": 4.0,
        "seed": seed,
        "verbosity": 0,
        "nthread": -1,
    }


def _rank_params(seed: int) -> dict:
    return {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@10",
        "eta": 0.03,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.85,
        "min_child_weight": 5.0,
        "gamma": 1.0,
        "alpha": 1.0,
        "lambda": 4.0,
        "seed": seed,
        "verbosity": 0,
        "nthread": -1,
    }


def _oof_ema_score(
    *,
    xgb,
    X_train_ema: np.ndarray,
    y_train: np.ndarray,
    train_groups: Sequence,
    seed: int,
    inner_groups: int,
) -> np.ndarray:
    splitter = CombinatorialPurgedGroupTimeSeriesSplit(
        train_groups,
        n_groups=min(int(inner_groups), len(set(train_groups))),
        test_group_count=1,
        embargo_groups=0,
    )
    params = _classifier_params(seed)
    oof = np.full(y_train.shape[0], np.nan, dtype=float)
    for inner_train_idx, inner_val_idx in splitter.split(X_train_ema, y_train):
        if np.unique(y_train[inner_train_idx]).size < 2:
            continue
        model = _train_xgb(
            xgb,
            X_train_ema[inner_train_idx],
            y_train[inner_train_idx],
            params=params,
            rounds=300,
        )
        oof[inner_val_idx] = _predict_xgb(xgb, model, X_train_ema[inner_val_idx])
    if np.isnan(oof).any():
        fallback = _train_xgb(xgb, X_train_ema, y_train, params=params, rounds=300)
        oof = np.where(np.isnan(oof), _predict_xgb(xgb, fallback, X_train_ema), oof)
    return oof


def _fit_ema_residual(
    *,
    xgb,
    X_train_ema: np.ndarray,
    y_train: np.ndarray,
    X_test_ema: np.ndarray,
    train_groups: Sequence,
    seed: int,
    inner_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    base_params = _classifier_params(seed)
    base_oof = _oof_ema_score(
        xgb=xgb,
        X_train_ema=X_train_ema,
        y_train=y_train,
        train_groups=train_groups,
        seed=seed,
        inner_groups=inner_groups,
    )
    base_model = _train_xgb(xgb, X_train_ema, y_train, params=base_params, rounds=300)
    base_train = _predict_xgb(xgb, base_model, X_train_ema)
    base_test = _predict_xgb(xgb, base_model, X_test_ema)
    residual = _train_xgb(
        xgb,
        X_train_ema,
        y_train,
        params=_residual_params(seed),
        rounds=150,
        base_margin=_logit(base_oof),
    )
    resid_train = _predict_xgb(xgb, residual, X_train_ema, base_margin=_logit(base_train))
    resid_test = _predict_xgb(xgb, residual, X_test_ema, base_margin=_logit(base_test))
    return resid_train, resid_test


def _load_legacy_labels(path: Path) -> pl.DataFrame:
    legacy = (
        pl.read_parquet(path)
        .filter(pl.col("portfolio_model").is_in(["Combined_Equal", "Combined_Frequency"]))
        .with_columns(pl.col("year_month").dt.date().alias("holding_month"))
        .select(["holding_month", "ticker", "n_models", "weight_normalized"])
        .group_by(["holding_month", "ticker"])
        .agg(
            pl.max("n_models").alias("legacy_n_models"),
            pl.max("weight_normalized").alias("legacy_weight_normalized"),
        )
        .with_columns(pl.lit(1).cast(pl.Int8).alias("legacy_selected"))
    )
    return legacy


def _append_legacy(frame: pl.DataFrame, legacy: pl.DataFrame) -> pl.DataFrame:
    return frame.join(legacy, on=["holding_month", "ticker"], how="left").with_columns(
        pl.col("legacy_selected").fill_null(0).cast(pl.Int8),
        pl.col("legacy_n_models").fill_null(0).cast(pl.Float64),
        pl.col("legacy_weight_normalized").fill_null(0.0).cast(pl.Float64),
    )


def _top_n(frame: pl.DataFrame, score_col: str, top_n: int) -> pl.DataFrame:
    return (
        frame.with_columns(pl.col(score_col).rank(method="ordinal", descending=True).over("holding_month").alias("rank"))
        .filter(pl.col("rank") <= int(top_n))
        .sort(["holding_month", "rank"])
    )


def _monthly_returns(selections: pl.DataFrame) -> pl.DataFrame:
    return (
        selections.group_by("holding_month")
        .agg(
            pl.mean("future_return").alias("portfolio_return"),
            pl.first("benchmark_future_return").alias("benchmark_return"),
            pl.mean("target_label").alias("hit_rate"),
            pl.mean("future_excess_return").alias("avg_excess_return"),
            pl.mean("legacy_selected").alias("legacy_overlap_rate"),
            pl.len().alias("n_positions"),
        )
        .rename({"holding_month": "year_month"})
        .with_columns((pl.col("portfolio_return") - pl.col("benchmark_return")).alias("active_return"))
        .sort("year_month")
    )


def _max_drawdown(returns: Sequence[float]) -> float:
    values = np.asarray(list(returns), dtype=float)
    if values.size == 0:
        return 0.0
    curve = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(curve)
    drawdown = curve / peak - 1.0
    return float(np.min(drawdown))


def _kpis(monthly: pl.DataFrame, model: str) -> dict:
    if monthly.is_empty():
        return {"model": model}
    active = monthly.get_column("active_return").to_numpy()
    return {
        "model": model,
        "months": monthly.height,
        "total_return": float((1.0 + monthly.get_column("portfolio_return")).product() - 1.0),
        "benchmark_return": float((1.0 + monthly.get_column("benchmark_return")).product() - 1.0),
        "active_compounded": float((1.0 + monthly.get_column("active_return")).product() - 1.0),
        "avg_monthly_active": float(np.mean(active)),
        "active_win_months": int(np.sum(active > 0.0)),
        "active_max_drawdown": _max_drawdown(active),
        "worst_active_month": float(np.min(active)),
        "best_active_month": float(np.max(active)),
        "avg_hit_rate": float(monthly.get_column("hit_rate").mean()),
        "avg_top10_excess": float(monthly.get_column("avg_excess_return").mean()),
        "avg_legacy_overlap": float(monthly.get_column("legacy_overlap_rate").mean()),
    }


def _group_sizes(frame: pl.DataFrame) -> list[int]:
    return frame.group_by("year_month", maintain_order=True).len().get_column("len").to_list()


def run_experiment(config: SignalCopyConfig) -> Path:
    xgb = load_xgboost()
    meta = json.loads((config.source_run / "metadata.json").read_text())
    full_features = [feature for feature in meta["features_used"]]
    model_frame = pl.read_parquet(config.source_run / "model_frame.parquet")
    ema_features = [
        column
        for column in full_features
        if column.startswith("ema_ratio_") or column.startswith("price_to_ema_")
    ]
    legacy = _load_legacy_labels(config.legacy_path)

    months = model_frame.select("year_month").unique().sort("year_month").get_column("year_month").to_list()
    windows = walk_forward_windows(
        months,
        min_train_months=config.min_train_months,
        val_months=config.val_months,
        test_months=config.test_months,
        step_months=config.step_months,
        max_windows=config.max_windows,
    )
    run_dir = config.output_dir / f"signal_copy_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames: list[pl.DataFrame] = []
    fold_rows: list[dict] = []

    for position, window in enumerate(windows, start=1):
        train_df = _append_legacy(
            filter_by_months(model_frame, window.train_months).filter(pl.col("future_excess_return").is_not_null()),
            legacy,
        )
        test_df = _append_legacy(
            filter_by_months(model_frame, window.test_months).filter(pl.col("future_excess_return").is_not_null()),
            legacy,
        )
        if train_df.height < 250 or test_df.height < 80:
            continue

        seed = config.seed + int(window.fold_index)
        y_train = _binary_target(train_df, config.threshold)
        y_test = _binary_target(test_df, config.threshold)
        if np.unique(y_train).size < 2:
            continue

        X_train_full = _matrix(train_df, full_features)
        X_test_full = _matrix(test_df, full_features)
        X_train_ema = _matrix(train_df, ema_features)
        X_test_ema = _matrix(test_df, ema_features)

        train_groups = train_df.get_column("year_month").to_list()
        ema_train_score, ema_test_score = _fit_ema_residual(
            xgb=xgb,
            X_train_ema=X_train_ema,
            y_train=y_train,
            X_test_ema=X_test_ema,
            train_groups=train_groups,
            seed=seed,
            inner_groups=config.inner_groups,
        )

        distill_available = train_df.filter(pl.col("holding_month") >= legacy.get_column("holding_month").min())
        y_distill = distill_available.get_column("legacy_selected").to_numpy().astype(np.int8)
        X_distill = _matrix(distill_available, full_features)
        if np.unique(y_distill).size < 2:
            distill_test_score = np.zeros(test_df.height, dtype=float)
        else:
            distill_model = _train_xgb(
                xgb,
                X_distill,
                y_distill,
                params=_classifier_params(seed),
                rounds=250,
            )
            distill_test_score = _predict_xgb(xgb, distill_model, X_test_full)

        blend_score = _zscore(ema_test_score) + _zscore(distill_test_score)

        y_reg = np.clip(train_df.get_column("future_excess_return").to_numpy(), -0.30, 0.30).astype(np.float32)
        reg_model = _train_xgb(xgb, X_train_full, y_reg, params=_regression_params(seed), rounds=250)
        regression_score = _predict_xgb(xgb, reg_model, X_test_full)

        train_rank_df = train_df.sort("year_month")
        X_train_rank = _matrix(train_rank_df, full_features)
        y_rank = _month_rank_target(train_rank_df, "future_excess_return")
        rank_model = _train_xgb(
            xgb,
            X_train_rank,
            y_rank,
            params=_rank_params(seed),
            rounds=250,
            group=_group_sizes(train_rank_df),
        )
        rank_score = _predict_xgb(xgb, rank_model, X_test_full)

        monotone = "(" + ",".join("1" if feature in ema_features else "0" for feature in full_features) + ")"
        monotone_model = _train_xgb(
            xgb,
            X_train_full,
            y_train,
            params=_classifier_params(seed, monotone_constraints=monotone),
            rounds=250,
        )
        monotone_score = _predict_xgb(xgb, monotone_model, X_test_full)

        two_stage_train = np.column_stack([X_train_full, ema_train_score])
        two_stage_test = np.column_stack([X_test_full, ema_test_score])
        two_stage_model = _train_xgb(
            xgb,
            two_stage_train,
            y_train,
            params=_classifier_params(seed),
            rounds=250,
        )
        two_stage_score = _predict_xgb(xgb, two_stage_model, two_stage_test)

        full_model = _train_xgb(xgb, X_train_full, y_train, params=_classifier_params(seed), rounds=250)
        full_score = _predict_xgb(xgb, full_model, X_test_full)
        ema_rank = _rank_pct(ema_test_score)
        gate_mask = ema_rank >= (1.0 - min(config.gate_n, test_df.height) / max(1, test_df.height))
        gated_score = np.where(gate_mask, full_score, -1e9)

        train_weight = np.ones(train_df.height, dtype=np.float32)
        train_weight += 3.0 * train_df.get_column("legacy_selected").to_numpy().astype(np.float32)
        train_weight += 2.0 * (train_df.get_column("future_excess_return").to_numpy() > config.threshold).astype(np.float32)
        weighted_model = _train_xgb(
            xgb,
            X_train_full,
            y_train,
            params=_classifier_params(seed),
            rounds=250,
            sample_weight=train_weight,
        )
        weighted_score = _predict_xgb(xgb, weighted_model, X_test_full)

        frame = test_df.select(IDENTITY_COLS + ["legacy_selected"]).with_columns(
            pl.Series("target_label", y_test, dtype=pl.Int8),
            pl.Series("distill_legacy", distill_test_score, dtype=pl.Float64),
            pl.Series("rank_pairwise", rank_score, dtype=pl.Float64),
            pl.Series("regression_excess", regression_score, dtype=pl.Float64),
            pl.Series("monotone_ema_full", monotone_score, dtype=pl.Float64),
            pl.Series("two_stage_ema_full", two_stage_score, dtype=pl.Float64),
            pl.Series("gated_full_after_ema", gated_score, dtype=pl.Float64),
            pl.Series("weighted_top_classifier", weighted_score, dtype=pl.Float64),
            pl.Series("ema_residual_benchmark", ema_test_score, dtype=pl.Float64),
            pl.Series("blend_ema_distill", blend_score, dtype=pl.Float64),
            pl.lit(window.fold_index).cast(pl.Int64).alias("fold"),
        )
        prediction_frames.append(frame)
        fold_rows.append(
            {
                "fold": window.fold_index,
                "test_month": str(window.test_months[0]),
                "test_rows": test_df.height,
                "test_positive_rate": float(np.mean(y_test)),
                "distill_auc": safe_auc(y_test, distill_test_score),
                "monotone_auc": safe_auc(y_test, monotone_score),
                "two_stage_auc": safe_auc(y_test, two_stage_score),
                "weighted_auc": safe_auc(y_test, weighted_score),
                "ema_residual_auc": safe_auc(y_test, ema_test_score),
            }
        )
        print(
            f"[{position}/{len(windows)}] {window.test_months[0]} "
            f"ema={fold_rows[-1]['ema_residual_auc']:.3f} "
            f"distill={fold_rows[-1]['distill_auc']:.3f} "
            f"two_stage={fold_rows[-1]['two_stage_auc']:.3f}"
        )

    predictions = pl.concat(prediction_frames, how="vertical")
    model_cols = [
        "distill_legacy",
        "rank_pairwise",
        "regression_excess",
        "monotone_ema_full",
        "two_stage_ema_full",
        "gated_full_after_ema",
        "weighted_top_classifier",
        "ema_residual_benchmark",
        "blend_ema_distill",
    ]
    monthly_frames: list[pl.DataFrame] = []
    kpi_rows: list[dict] = []
    for model_name in model_cols:
        selections = _top_n(predictions, model_name, config.top_n).with_columns(pl.lit(model_name).alias("model"))
        monthly = _monthly_returns(selections).with_columns(pl.lit(model_name).alias("model"))
        selections.write_parquet(run_dir / f"selections_{model_name}.parquet")
        monthly.write_parquet(run_dir / f"monthly_{model_name}.parquet")
        monthly_frames.append(monthly)
        kpi_rows.append(_kpis(monthly, model_name))

    all_monthly = pl.concat(monthly_frames, how="vertical")
    kpis = pl.DataFrame(kpi_rows).sort("active_compounded", descending=True)
    fold_metrics = pl.DataFrame(fold_rows)

    predictions.write_parquet(run_dir / "predictions.parquet")
    all_monthly.write_parquet(run_dir / "monthly_returns.parquet")
    kpis.write_parquet(run_dir / "kpis.parquet")
    kpis.write_csv(run_dir / "kpis.csv")
    fold_metrics.write_parquet(run_dir / "fold_metrics.parquet")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": config.__dict__ | {
                    "source_run": str(config.source_run),
                    "legacy_path": str(config.legacy_path),
                    "output_dir": str(config.output_dir),
                },
                "full_features": full_features,
                "ema_features": ema_features,
                "models": model_cols,
                "primary_metric": "active_compounded",
                "metric_notes": {
                    "active_compounded": "Compounded top-N active return versus SPY over the walk-forward test months.",
                    "avg_top10_excess": "Average future excess return of selected top-N names.",
                    "avg_legacy_overlap": "Average share of selected names also selected by legacy for that holding month.",
                    "active_max_drawdown": "Max drawdown of compounded active monthly returns.",
                },
            },
            indent=2,
            default=str,
        )
    )
    print(f"RUN_DIR={run_dir}")
    print(kpis)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test signal-copy boosting variants against EMA and legacy baselines.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--gate-n", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        SignalCopyConfig(
            source_run=args.source_run,
            legacy_path=args.legacy_path,
            output_dir=args.output_dir,
            max_windows=args.max_windows,
            top_n=args.top_n,
            gate_n=args.gate_n,
            threshold=args.threshold,
        )
    )


if __name__ == "__main__":
    main()
