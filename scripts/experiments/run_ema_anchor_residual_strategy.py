from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
from run_signal_copy_models import DEFAULT_SOURCE_RUN  # noqa: E402
from run_tradable_ema_regression_optuna import (  # noqa: E402
    _add_cross_sectional_features,
    _ema_base_features,
)
from run_tradable_ema_regression_trading_backtest import (  # noqa: E402
    DEFAULT_LEGACY_MONTHLY_RETURNS,
    build_spy_curve,
    load_legacy_curves,
)

from alpharank.backtest.application import compare_backtest_curves
from alpharank.backtest.mlcraft_adapter import (
    ensure_mlcraft_importable,
    to_mlcraft_model_and_fit_params,
)
from alpharank.backtest.portfolio import compute_monthly_portfolio_returns, select_top_n
from alpharank.backtest.time_folds import filter_by_months, walk_forward_windows
from alpharank.utils.xgboost_runtime import load_xgboost

DEFAULT_LEGACY_DETAILED_RETURNS = Path("outputs/2026-06-07/legacy_detailed_returns_polars.parquet")
DEFAULT_PRICE_VS_INDEX = Path("outputs/checkpoints_open_source_20260607/polars_final_price_vs_index.parquet")


@dataclass(frozen=True)
class EmaAnchorResidualConfig:
    source_run: Path = DEFAULT_SOURCE_RUN
    legacy_monthly_returns: Path = DEFAULT_LEGACY_MONTHLY_RETURNS
    legacy_detailed_returns: Path = DEFAULT_LEGACY_DETAILED_RETURNS
    price_vs_index: Path = DEFAULT_PRICE_VS_INDEX
    output_dir: Path = Path("outputs")
    anchor_mode: str = "legacy_exact_dominant"
    residual_mode: str = "init_score"
    fixed_ema_short: int | None = None
    fixed_ema_long: int | None = None
    target_clip: float = 0.30
    min_train_months: int = 168
    val_months: int = 12
    test_months: int = 1
    step_months: int = 1
    max_windows: int = 999
    ema_selection_top_k: int = 20
    top_n_values: tuple[int, ...] = (5, 7, 10, 20, 30, 50)
    risk_free_rate: float = 0.02
    seed: int = 42


BASE_MODEL_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 220,
    "learning_rate": 0.025,
    "max_depth": 2,
    "subsample": 0.80,
    "colsample_bytree": 0.90,
    "min_child_weight": 8.0,
    "gamma": 1.0,
    "reg_alpha": 1.0,
    "reg_lambda": 6.0,
    "n_jobs": -1,
}

RESIDUAL_MODEL_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 180,
    "learning_rate": 0.02,
    "max_depth": 2,
    "subsample": 0.75,
    "colsample_bytree": 0.70,
    "min_child_weight": 10.0,
    "gamma": 1.5,
    "reg_alpha": 2.0,
    "reg_lambda": 10.0,
    "n_jobs": -1,
}


def _feature_family(feature: str) -> list[str]:
    return [
        feature,
        f"{feature}_rank_month",
        f"{feature}_z_month",
        f"{feature}_top25_flag",
        f"{feature}_bottom25_flag",
    ]


def _legacy_ema_feature_name(n_short: int, n_long: int) -> str:
    return f"legacy_ema_ratio_short{int(n_short)}_long{int(n_long)}"


def _load_legacy_atomic_configs(path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    legacy = (
        pl.read_parquet(path)
        .with_columns(
            pl.col("year_month").dt.date().alias("holding_month"),
            pl.col("ticker").cast(pl.Utf8),
            pl.col("n_short").cast(pl.Int32),
            pl.col("n_long").cast(pl.Int32),
        )
        .filter(pl.col("portfolio_model").str.starts_with("Legacy_Optuna"))
    )
    pairs = legacy.select(["n_short", "n_long"]).drop_nulls().unique().sort(["n_long", "n_short"])
    dominant = (
        legacy.group_by(["holding_month", "selected_model", "n_short", "n_long"])
        .agg(pl.len().alias("legacy_atomic_rows"))
        .sort(["holding_month", "legacy_atomic_rows", "selected_model"], descending=[False, True, False])
        .group_by("holding_month", maintain_order=True)
        .head(1)
        .with_columns(
            pl.struct(["n_short", "n_long"])
            .map_elements(
                lambda row: _legacy_ema_feature_name(int(row["n_short"]), int(row["n_long"])),
                return_dtype=pl.Utf8,
            )
            .alias("legacy_primary_ema_feature")
        )
        .select(["holding_month", "selected_model", "n_short", "n_long", "legacy_atomic_rows", "legacy_primary_ema_feature"])
    )
    return pairs, dominant


def _fixed_pair_frame(config: EmaAnchorResidualConfig) -> pl.DataFrame:
    if config.fixed_ema_short is None or config.fixed_ema_long is None:
        raise ValueError("fixed_exact anchor mode requires fixed_ema_short and fixed_ema_long.")
    return pl.DataFrame(
        {
            "n_short": [int(config.fixed_ema_short)],
            "n_long": [int(config.fixed_ema_long)],
        },
        schema={"n_short": pl.Int32, "n_long": pl.Int32},
    )


def _exact_legacy_ema_features(price_path: Path, keys: pl.DataFrame, pairs: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    prices = (
        pl.read_parquet(price_path)
        .select(["ticker", "date", "close_vs_index"])
        .with_columns(pl.col("ticker").cast(pl.Utf8), pl.col("date").cast(pl.Datetime))
        .drop_nulls(["ticker", "date", "close_vs_index"])
        .sort(["ticker", "date"])
    )
    base = keys.select(["holding_month", "ticker"]).unique()
    feature_names: list[str] = []
    for pair in pairs.iter_rows(named=True):
        n_short = int(pair["n_short"])
        n_long = int(pair["n_long"])
        feature = _legacy_ema_feature_name(n_short, n_long)
        feature_names.append(feature)
        monthly = (
            prices.with_row_index("_row_idx")
            .with_columns(
                pl.col("close_vs_index").ewm_mean(span=n_short, adjust=False).over("ticker").alias("_ema_short"),
                pl.col("close_vs_index").ewm_mean(span=n_long, adjust=False).over("ticker").alias("_ema_long"),
                (pl.col("_row_idx").rank(method="ordinal").over("ticker") - 1).alias("_obs_count"),
                pl.col("date").dt.truncate("1mo").dt.date().alias("_signal_month"),
            )
            .filter(pl.col("_obs_count") >= n_long)
            .with_columns((pl.col("_ema_short") / pl.col("_ema_long")).alias(feature))
            .drop_nulls(feature)
            .group_by(["_signal_month", "ticker"])
            .agg(pl.col("date").last(), pl.col(feature).last())
            .with_columns(pl.col("_signal_month").dt.offset_by("1mo").alias("holding_month"))
            .select(["holding_month", "ticker", feature])
        )
        base = base.join(monthly, on=["holding_month", "ticker"], how="left")
    return base, feature_names


def _dynamic_primary_expr(candidates: Sequence[str], selector_col: str, output_col: str) -> pl.Expr:
    expr: pl.Expr | None = None
    for feature in candidates:
        branch = pl.when(pl.col(selector_col) == feature).then(pl.col(feature))
        expr = branch if expr is None else expr.when(pl.col(selector_col) == feature).then(pl.col(feature))
    if expr is None:
        raise ValueError("No exact Legacy EMA candidates available.")
    return expr.otherwise(None).alias(output_col)


def _load_frame(config: EmaAnchorResidualConfig) -> tuple[pl.DataFrame, list[str], list[str], list[str]]:
    metadata = json.loads((config.source_run / "metadata.json").read_text(encoding="utf-8"))
    source_features = list(metadata["features_used"])
    available_ema_features = _ema_base_features(source_features)
    if not available_ema_features:
        raise ValueError("No EMA features found in source metadata.")

    frame = pl.read_parquet(config.source_run / "model_frame.parquet").with_columns(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
    )
    frame = frame.filter(pl.col("future_return").is_not_null(), pl.col("future_excess_return").is_not_null())

    ema_features = available_ema_features
    if config.anchor_mode in {"legacy_exact_dominant", "validation_exact", "fixed_exact"}:
        if config.anchor_mode == "fixed_exact":
            pairs = _fixed_pair_frame(config)
            dominant = pl.DataFrame()
        else:
            pairs, dominant = _load_legacy_atomic_configs(config.legacy_detailed_returns)
        exact_ema, exact_ema_features = _exact_legacy_ema_features(
            config.price_vs_index,
            frame.select(["holding_month", "ticker"]),
            pairs,
        )
        frame = frame.join(exact_ema, on=["holding_month", "ticker"], how="left")
        source_features = source_features + exact_ema_features
        ema_features = exact_ema_features
        if config.anchor_mode == "legacy_exact_dominant":
            frame = frame.join(dominant, on="holding_month", how="left")
            primary_col = "legacy_exact_primary_mtr"
            frame = frame.with_columns(_dynamic_primary_expr(exact_ema_features, "legacy_primary_ema_feature", primary_col))
            source_features.append(primary_col)
            ema_features = [primary_col]
        elif config.anchor_mode == "fixed_exact":
            ema_features = exact_ema_features

    frame, enriched_features = _add_cross_sectional_features(frame, source_features, prefix="anchor_all")
    return frame, source_features, ema_features, enriched_features


def _matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    return frame.select(list(features)).fill_null(0.0).to_numpy().astype(np.float32)


def _target(frame: pl.DataFrame, clip: float) -> np.ndarray:
    return np.clip(frame.get_column("future_excess_return").to_numpy(), -clip, clip).astype(np.float32)


def _fit_mlcraft_regressor(*, params: dict[str, Any], X: np.ndarray, y: np.ndarray, seed: int):
    ensure_mlcraft_importable()
    from mlcraft.core.task import TaskSpec
    from mlcraft.models.factory import ModelFactory

    model_params, fit_params = to_mlcraft_model_and_fit_params(params)
    fit_params = dict(fit_params)
    if "num_boost_round" not in fit_params:
        fit_params["num_boost_round"] = int(params.get("n_estimators", 200))
    model = ModelFactory.create(
        "xgboost",
        task_spec=TaskSpec(task_type="regression"),
        model_params=model_params,
        fit_params=fit_params,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def _xgb_regression_params(params: dict[str, Any], seed: int) -> dict[str, Any]:
    converted: dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": int(seed),
        "verbosity": 0,
    }
    for key, value in params.items():
        if key in {"objective", "eval_metric", "n_estimators"}:
            continue
        if key == "learning_rate":
            converted["eta"] = value
        elif key == "reg_alpha":
            converted["alpha"] = value
        elif key == "reg_lambda":
            converted["lambda"] = value
        elif key == "n_jobs":
            converted["nthread"] = value
        else:
            converted[key] = value
    return converted


def _fit_xgb_regressor_with_base_margin(
    *,
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    base_margin: np.ndarray,
    seed: int,
):
    xgb = load_xgboost()
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_base_margin(np.asarray(base_margin, dtype=np.float32))
    return xgb.train(
        params=_xgb_regression_params(params, seed),
        dtrain=dtrain,
        num_boost_round=int(params.get("n_estimators", 200)),
        verbose_eval=False,
    )


def _predict_xgb_with_base_margin(model: Any, X: np.ndarray, base_margin: np.ndarray) -> np.ndarray:
    xgb = load_xgboost()
    dtest = xgb.DMatrix(X)
    dtest.set_base_margin(np.asarray(base_margin, dtype=np.float32))
    return np.asarray(model.predict(dtest), dtype=float).reshape(-1)


def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    pred = model.predict(X)
    if isinstance(pred, tuple):
        pred = pred[0]
    return np.asarray(pred, dtype=float).reshape(-1)


def _select_primary_ema(val_df: pl.DataFrame, ema_features: Sequence[str], top_k: int) -> tuple[str, float]:
    rows: list[dict[str, Any]] = []
    for feature in ema_features:
        rank_col = f"{feature}_rank_month"
        if rank_col not in val_df.columns:
            continue
        top = (
            val_df.with_columns(pl.col(rank_col).rank(method="ordinal", descending=True).over("year_month").alias("_rank"))
            .filter(pl.col("_rank") <= int(top_k))
            .group_by("year_month")
            .agg(pl.mean("future_excess_return").alias("_monthly_excess"))
        )
        if top.is_empty():
            continue
        rows.append({"primary_ema": feature, "validation_topk_excess": float(top.get_column("_monthly_excess").mean())})
    if not rows:
        raise ValueError("Unable to select a primary EMA on validation.")
    best = max(rows, key=lambda row: row["validation_topk_excess"])
    return str(best["primary_ema"]), float(best["validation_topk_excess"])


def _prediction_frame(
    test_df: pl.DataFrame,
    *,
    fold: str,
    primary_ema: str,
    base_prediction: np.ndarray,
    residual_prediction: np.ndarray,
) -> pl.DataFrame:
    return test_df.select(
        [
            "ticker",
            "year_month",
            "decision_month",
            "decision_asof_date",
            "holding_month",
            "future_return",
            "benchmark_future_return",
            "future_excess_return",
        ]
    ).with_columns(
        pl.lit(fold).alias("fold"),
        pl.lit(primary_ema).alias("primary_ema"),
        pl.Series("ema_anchor_prediction", base_prediction, dtype=pl.Float64),
        pl.Series("residual_prediction", residual_prediction, dtype=pl.Float64),
        pl.Series("ema_anchor_residual_prediction", base_prediction + residual_prediction, dtype=pl.Float64),
        (pl.col("future_excess_return") > 0.0).cast(pl.Int8).alias("target_label"),
    )


def _run_scenario(predictions: pl.DataFrame, score_col: str, name: str, top_n: int) -> dict[str, pl.DataFrame]:
    application = predictions.with_columns(pl.col(score_col).alias("prediction"))
    selections = select_top_n(application, top_n=top_n)
    monthly = compute_monthly_portfolio_returns(selections)
    return {
        "monthly": monthly.with_columns(pl.lit(name).alias("model")),
        "selections": selections.with_columns(pl.lit(name).alias("model")),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x_m = x[mask]
    y_m = y[mask]
    if np.std(x_m) < 1e-12 or np.std(y_m) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_m, y_m)[0, 1])


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _prediction_metrics(predictions: pl.DataFrame, score_col: str, label: str, top_n_values: Sequence[int]) -> pl.DataFrame:
    y = predictions.get_column("future_excess_return").to_numpy().astype(float)
    pred = predictions.get_column(score_col).to_numpy().astype(float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y_m = y[mask]
    pred_m = pred[mask]
    residual = y_m - pred_m
    sst = float(np.sum((y_m - np.mean(y_m)) ** 2)) if y_m.size else float("nan")
    sse = float(np.sum(residual**2)) if y_m.size else float("nan")
    base_rows = [
        {
            "model": label,
            "metric": "rmse",
            "value": float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan"),
        },
        {"model": label, "metric": "mae", "value": float(np.mean(np.abs(residual))) if residual.size else float("nan")},
        {"model": label, "metric": "r2", "value": float(1.0 - sse / sst) if sst and sst > 1e-12 else float("nan")},
        {"model": label, "metric": "pearson", "value": _safe_corr(pred_m, y_m)},
        {"model": label, "metric": "spearman", "value": _safe_corr(_rank_average(pred_m), _rank_average(y_m))},
    ]
    for top_n in top_n_values:
        selected = (
            predictions.with_columns(pl.col(score_col).rank(method="ordinal", descending=True).over("year_month").alias("_rank"))
            .filter(pl.col("_rank") <= int(top_n))
            .group_by("year_month")
            .agg(
                pl.mean("future_excess_return").alias("avg_excess"),
                (pl.col("future_excess_return") > 0.0).mean().alias("hit_rate"),
            )
        )
        if selected.is_empty():
            continue
        base_rows.extend(
            [
                {
                    "model": label,
                    "metric": f"top{top_n}_avg_monthly_excess",
                    "value": float(selected.get_column("avg_excess").mean()),
                },
                {"model": label, "metric": f"top{top_n}_hit_rate", "value": float(selected.get_column("hit_rate").mean())},
            ]
        )
    return pl.DataFrame(base_rows)


def _write_report(
    run_dir: Path,
    *,
    comparison_metrics: pl.DataFrame,
    prediction_metrics: pl.DataFrame,
    fold_metrics: pl.DataFrame,
    config: EmaAnchorResidualConfig,
) -> None:
    rows = comparison_metrics.sort("Total Return", descending=True).to_dicts()
    metric_rows = prediction_metrics.to_dicts()
    lines = [
        "# EMA anchor residual strategy",
        "",
        "But: tester une strategie en deux etages.",
        "",
        "1. Construire une EMA primaire point-in-time.",
        "2. Entrainer un boosting mlcraft sur le futur rendement relatif avec seulement cette EMA.",
        "3. Entrainer un deuxieme boosting sur le residu avec toutes les autres variables disponibles.",
        "4. Comparer le top K du modele EMA seul au top K du modele EMA + residu.",
        "",
        "Les tickers selectionnes par Legacy ne sont pas utilises comme feature ou objectif.",
        "Quand une configuration EMA Legacy est fournie, elle sert seulement a construire l'ancre primaire.",
        "Legacy sert ensuite de benchmark de performance final.",
        "",
        f"Source run: `{config.source_run}`",
        f"Mode EMA primaire: `{config.anchor_mode}`.",
        f"Mode residu: `{config.residual_mode}`.",
        f"Target: `future_excess_return` clippe a +/-{config.target_clip:.2f}.",
    ]
    if config.anchor_mode == "legacy_exact_dominant":
        lines.extend(
            [
                "EMA primaire: couple exact `n_short/n_long` du modele Legacy atomique dominant du mois, calcule depuis `close_vs_index`.",
                f"Legacy detail: `{config.legacy_detailed_returns}`",
                f"Prix relatifs: `{config.price_vs_index}`",
            ]
        )
    elif config.anchor_mode == "fixed_exact":
        lines.extend(
            [
                f"EMA primaire fixe: `n_short={config.fixed_ema_short}`, `n_long={config.fixed_ema_long}`.",
                f"Prix relatifs: `{config.price_vs_index}`",
            ]
        )
    else:
        lines.append(f"Selection EMA validation: top {config.ema_selection_top_k} moyen en futur rendement relatif.")
    if config.residual_mode == "init_score":
        lines.append(
            "Le second etage est entraine avec l'init score XGBoost `base_margin` du modele EMA primaire."
        )
    lines.extend(
        [
        "",
        "## Backtest",
        "",
        "| modele | total return | CAGR | Sharpe | max drawdown | vol mensuelle | mois positifs |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['model']}` | {row['Total Return'] * 100:.1f}% | {row['CAGR'] * 100:.1f}% | "
            f"{row['Sharpe Ratio']:.2f} | {row['Max Drawdown'] * 100:.1f}% | "
            f"{row['Monthly Volatility'] * 100:.1f}% | {row['Positive Periods %'] * 100:.1f}% |"
        )
    lines.extend(["", "## Metriques prediction test", "", "| modele | metrique | valeur |", "|---|---|---:|"])
    for row in metric_rows:
        value = row["value"]
        if row["metric"].endswith("hit_rate") or row["metric"].endswith("excess"):
            formatted = f"{value * 100:.2f}%"
        else:
            formatted = f"{value:.4f}"
        lines.append(f"| `{row['model']}` | `{row['metric']}` | {formatted} |")

    ema_counts = (
        fold_metrics.group_by("primary_ema")
        .agg(pl.len().alias("folds"), pl.mean("validation_topk_excess").alias("avg_validation_topk_excess"))
        .sort("folds", descending=True)
        .to_dicts()
    )
    lines.extend(["", "## EMA primaires selectionnees", "", "| EMA | folds | validation topK excess moyen |", "|---|---:|---:|"])
    for row in ema_counts:
        lines.append(f"| `{row['primary_ema']}` | {row['folds']} | {row['avg_validation_topk_excess'] * 100:.2f}% |")

    lines.extend(
        [
            "",
            "## Lecture",
            "",
            "- `ema_anchor_topK` = top K du boosting entraine uniquement sur l'EMA primaire.",
            "- `ema_anchor_residual_topK` = top K du score base EMA + prediction du residu par les autres variables.",
            "- Les metriques prediction sont calculees uniquement sur les lignes de test out-of-sample.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: EmaAnchorResidualConfig) -> Path:
    run_dir = config.output_dir / f"ema_anchor_residual_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    frame, source_features, ema_features, _ = _load_frame(config)
    months = frame.select("year_month").unique().sort("year_month").get_column("year_month").to_list()
    windows = walk_forward_windows(
        months,
        min_train_months=config.min_train_months,
        val_months=config.val_months,
        test_months=config.test_months,
        step_months=config.step_months,
        max_windows=config.max_windows,
    )
    prediction_frames: list[pl.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for position, window in enumerate(windows, start=1):
        fold = f"fold_{window.fold_index:03d}"
        train_df = filter_by_months(frame, window.train_months)
        val_df = filter_by_months(frame, window.val_months)
        test_df = filter_by_months(frame, window.test_months)
        if train_df.is_empty() or val_df.is_empty() or test_df.is_empty():
            continue

        primary_ema, validation_topk_excess = _select_primary_ema(val_df, ema_features, config.ema_selection_top_k)
        base_features = _feature_family(primary_ema)
        residual_source_features = [feature for feature in source_features if feature != primary_ema]
        residual_features: list[str] = []
        for feature in residual_source_features:
            residual_features.extend(_feature_family(feature))
        residual_features = [feature for feature in residual_features if feature in train_df.columns]

        seed = config.seed + position
        y_train = _target(train_df, config.target_clip)
        base_model = _fit_mlcraft_regressor(
            params=BASE_MODEL_PARAMS,
            X=_matrix(train_df, base_features),
            y=y_train,
            seed=seed,
        )
        base_train_pred = _predict(base_model, _matrix(train_df, base_features))
        base_test_pred = _predict(base_model, _matrix(test_df, base_features))
        if config.residual_mode == "target_residual":
            residual_target = (y_train - base_train_pred).astype(np.float32)
            residual_model = _fit_mlcraft_regressor(
                params=RESIDUAL_MODEL_PARAMS,
                X=_matrix(train_df, residual_features),
                y=residual_target,
                seed=seed,
            )
            residual_test_pred = _predict(residual_model, _matrix(test_df, residual_features))
            final_test_pred = base_test_pred + residual_test_pred
        elif config.residual_mode == "init_score":
            residual_model = _fit_xgb_regressor_with_base_margin(
                params=RESIDUAL_MODEL_PARAMS,
                X=_matrix(train_df, residual_features),
                y=y_train,
                base_margin=base_train_pred,
                seed=seed,
            )
            final_test_pred = _predict_xgb_with_base_margin(
                residual_model,
                _matrix(test_df, residual_features),
                base_margin=base_test_pred,
            )
            residual_test_pred = final_test_pred - base_test_pred
        else:
            raise ValueError(f"Unsupported residual_mode: {config.residual_mode!r}")
        prediction_frames.append(
            _prediction_frame(
                test_df,
                fold=fold,
                primary_ema=primary_ema,
                base_prediction=base_test_pred,
                residual_prediction=residual_test_pred,
            )
            .with_columns(pl.Series("ema_anchor_residual_prediction", final_test_pred, dtype=pl.Float64))
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_start": str(min(window.train_months)),
                "train_end": str(max(window.train_months)),
                "val_start": str(min(window.val_months)),
                "val_end": str(max(window.val_months)),
                "test_start": str(min(window.test_months)),
                "test_end": str(max(window.test_months)),
                "primary_ema": primary_ema,
                "validation_topk_excess": validation_topk_excess,
                "residual_mode": config.residual_mode,
                "base_feature_count": len(base_features),
                "residual_feature_count": len(residual_features),
            }
        )
        print(f"{fold}: test={window.test_months[0]} primary={primary_ema} val_topk={validation_topk_excess:.4f}", flush=True)

    predictions = pl.concat(prediction_frames, how="vertical") if prediction_frames else pl.DataFrame()
    fold_metrics = pl.DataFrame(fold_rows)

    scenarios: list[dict[str, pl.DataFrame]] = []
    for top_n in config.top_n_values:
        scenarios.append(_run_scenario(predictions, "ema_anchor_prediction", f"ema_anchor_top{top_n}", top_n))
        scenarios.append(_run_scenario(predictions, "ema_anchor_residual_prediction", f"ema_anchor_residual_top{top_n}", top_n))
    monthly_returns = pl.concat([scenario["monthly"] for scenario in scenarios], how="vertical")
    selections = pl.concat([scenario["selections"] for scenario in scenarios], how="diagonal_relaxed")
    months_out = predictions.select(pl.col("holding_month").alias("year_month")).unique().sort("year_month").get_column("year_month").to_list()

    curves = {
        scenario["monthly"].get_column("model")[0]: scenario["monthly"].select(
            "year_month",
            pl.col("portfolio_return").alias("monthly_return"),
            pl.col("n_positions").alias("n"),
        )
        for scenario in scenarios
    }
    curves["SPY"] = build_spy_curve(predictions)
    curves.update(load_legacy_curves(config.legacy_monthly_returns, months_out))
    comparison = compare_backtest_curves(
        curves,
        output_path=run_dir / "comparison.html",
        title="EMA anchor residual strategy vs Legacy",
        risk_free_rate=config.risk_free_rate,
    )

    prediction_metrics = pl.concat(
        [
            _prediction_metrics(predictions, "ema_anchor_prediction", "ema_anchor", config.top_n_values),
            _prediction_metrics(predictions, "ema_anchor_residual_prediction", "ema_anchor_residual", config.top_n_values),
        ],
        how="vertical",
    )

    predictions.write_parquet(run_dir / "predictions.parquet")
    selections.write_parquet(run_dir / "selections.parquet")
    monthly_returns.write_parquet(run_dir / "monthly_returns.parquet")
    fold_metrics.write_csv(run_dir / "fold_metrics.csv")
    prediction_metrics.write_csv(run_dir / "prediction_metrics.csv")
    comparison.metrics.write_csv(run_dir / "comparison_metrics.csv")
    comparison.annual_returns.write_csv(run_dir / "annual_returns.csv")
    comparison.correlation_matrix.write_csv(run_dir / "correlation_matrix.csv")
    comparison.worst_periods.write_csv(run_dir / "worst_periods.csv")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                "source_features": source_features,
                "ema_candidates": ema_features,
                "base_model_params": BASE_MODEL_PARAMS,
                "residual_model_params": RESIDUAL_MODEL_PARAMS,
                "residual_training": (
                    "XGBoost base_margin/init_score through direct xgboost API because current mlcraft XGBoost wrapper "
                    "does not expose base_margin."
                    if config.residual_mode == "init_score"
                    else "mlcraft residual target regression"
                ),
                "months": len(months_out),
                "start_month": str(min(months_out)) if months_out else None,
                "end_month": str(max(months_out)) if months_out else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(
        run_dir,
        comparison_metrics=comparison.metrics,
        prediction_metrics=prediction_metrics,
        fold_metrics=fold_metrics,
        config=config,
    )
    print(f"RUN_DIR={run_dir}")
    print(comparison.metrics.sort("Total Return", descending=True).head(20))
    return run_dir


def _parse_args() -> EmaAnchorResidualConfig:
    parser = argparse.ArgumentParser(description="Train EMA-anchor plus residual boosting strategy.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--legacy-monthly-returns", type=Path, default=DEFAULT_LEGACY_MONTHLY_RETURNS)
    parser.add_argument("--legacy-detailed-returns", type=Path, default=DEFAULT_LEGACY_DETAILED_RETURNS)
    parser.add_argument("--price-vs-index", type=Path, default=DEFAULT_PRICE_VS_INDEX)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--anchor-mode",
        choices=["legacy_exact_dominant", "validation_exact", "validation_available", "fixed_exact"],
        default="legacy_exact_dominant",
    )
    parser.add_argument("--residual-mode", choices=["init_score", "target_residual"], default="init_score")
    parser.add_argument("--fixed-ema-short", type=int, default=None)
    parser.add_argument("--fixed-ema-long", type=int, default=None)
    parser.add_argument(
        "--fixed-ema-pair",
        type=str,
        default=None,
        help="Legacy text order n_long-n_short, for example 34-7 means n_long=34 and n_short=7.",
    )
    parser.add_argument("--target-clip", type=float, default=0.30)
    parser.add_argument("--min-train-months", type=int, default=168)
    parser.add_argument("--val-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=999)
    parser.add_argument("--ema-selection-top-k", type=int, default=20)
    parser.add_argument("--top-n", type=int, nargs="*", default=[5, 7, 10, 20, 30, 50])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    fixed_ema_short = args.fixed_ema_short
    fixed_ema_long = args.fixed_ema_long
    if args.fixed_ema_pair:
        raw_long, raw_short = args.fixed_ema_pair.replace("/", "-").split("-", maxsplit=1)
        fixed_ema_long = int(float(raw_long))
        fixed_ema_short = int(float(raw_short))
    return EmaAnchorResidualConfig(
        source_run=args.source_run,
        legacy_monthly_returns=args.legacy_monthly_returns,
        legacy_detailed_returns=args.legacy_detailed_returns,
        price_vs_index=args.price_vs_index,
        output_dir=args.output_dir,
        anchor_mode=args.anchor_mode,
        residual_mode=args.residual_mode,
        fixed_ema_short=fixed_ema_short,
        fixed_ema_long=fixed_ema_long,
        target_clip=args.target_clip,
        min_train_months=args.min_train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        step_months=args.step_months,
        max_windows=args.max_windows,
        ema_selection_top_k=args.ema_selection_top_k,
        top_n_values=tuple(args.top_n),
        seed=args.seed,
    )


if __name__ == "__main__":
    run(_parse_args())
