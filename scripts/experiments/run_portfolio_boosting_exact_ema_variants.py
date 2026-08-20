from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import optuna
import polars as pl
from optuna.samplers import TPESampler
from run_portfolio_boosting_exact_ema_rank import (  # noqa: E402
    PortfolioBoostingExactEmaRankConfig,
    _load_exact_ema_rank_frame,
    _prediction_metrics,
)
from run_portfolio_boosting_rank_regression import (  # noqa: E402
    BASE_PARAMS as RANK_BASE_PARAMS,
)
from run_portfolio_boosting_rank_regression import (
    SEARCH_SPACE as RANK_SEARCH_SPACE,
)
from run_portfolio_boosting_rank_regression import (
    _base_trial_params,
    _fit_mlcraft_regressor,
    _matrix,
    _predict,
    _run_model_scenario,
    _sample_params,
    _target,
    _write_warm_start_candidates,
)
from run_portfolio_boosting_top_return_classifier import (  # noqa: E402
    BASE_PARAMS as CLASSIFIER_BASE_PARAMS,
)
from run_portfolio_boosting_top_return_classifier import (
    _fit_mlcraft_classifier,
)
from run_tradable_ema_regression_trading_backtest import (  # noqa: E402
    DEFAULT_LEGACY_MONTHLY_RETURNS,
    build_spy_curve,
    load_legacy_curves,
)

from alpharank.backtest.application import compare_backtest_curves
from alpharank.backtest.mlcraft_adapter import to_mlcraft_search_space
from alpharank.backtest.portfolio import select_top_n
from alpharank.backtest.time_folds import filter_by_months, walk_forward_windows
from alpharank.utils.xgboost_runtime import load_xgboost


@dataclass(frozen=True)
class PortfolioBoostingExactEmaVariantsConfig:
    output_dir: Path = Path("outputs")
    legacy_monthly_returns: Path = DEFAULT_LEGACY_MONTHLY_RETURNS
    n_trials: int = 2
    startup_trials: int = 1
    max_windows: int = 999
    min_train_months: int = 168
    val_months: int = 12
    test_months: int = 1
    step_months: int = 1
    objective_top_k: int = 10
    top_n_values: tuple[int, ...] = (5, 7, 10, 20, 30, 50)
    ranker_objective: str = "rank:pairwise"
    return_clip: float = 0.30
    positive_quantile: float = 0.90
    robust_vol_penalty: float = 0.35
    robust_dd_penalty: float = 0.25
    robust_gap_penalty: float = 0.10
    risk_free_rate: float = 0.02
    seed: int = 42


XGB_RANK_BASE_PARAMS: dict[str, Any] = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg@10",
    "eta": 0.03,
    "max_depth": 3,
    "subsample": 0.80,
    "colsample_bytree": 0.85,
    "min_child_weight": 5.0,
    "gamma": 1.0,
    "alpha": 1.0,
    "lambda": 4.0,
    "verbosity": 0,
    "nthread": -1,
}

XGB_RANK_SEARCH_SPACE: dict[str, tuple[str, float, float]] = {
    "num_boost_round": ("int", 80, 260),
    "eta": ("loguniform", 0.006, 0.08),
    "max_depth": ("int", 1, 4),
    "subsample": ("float", 0.55, 0.95),
    "colsample_bytree": ("float", 0.45, 0.95),
    "min_child_weight": ("float", 3.0, 30.0),
    "gamma": ("float", 0.0, 10.0),
    "alpha": ("float", 0.0, 12.0),
    "lambda": ("float", 1.0, 20.0),
}


def _xgb_sample_params(trial: optuna.Trial) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, (kind, low, high) in XGB_RANK_SEARCH_SPACE.items():
        if kind == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        elif kind == "loguniform":
            params[name] = trial.suggest_float(name, float(low), float(high), log=True)
        else:
            params[name] = trial.suggest_float(name, float(low), float(high))
    return params


def _xgb_base_trial_params() -> dict[str, Any]:
    return {
        "num_boost_round": 180,
        "eta": XGB_RANK_BASE_PARAMS["eta"],
        "max_depth": XGB_RANK_BASE_PARAMS["max_depth"],
        "subsample": XGB_RANK_BASE_PARAMS["subsample"],
        "colsample_bytree": XGB_RANK_BASE_PARAMS["colsample_bytree"],
        "min_child_weight": XGB_RANK_BASE_PARAMS["min_child_weight"],
        "gamma": XGB_RANK_BASE_PARAMS["gamma"],
        "alpha": XGB_RANK_BASE_PARAMS["alpha"],
        "lambda": XGB_RANK_BASE_PARAMS["lambda"],
    }


def _group_sizes(frame: pl.DataFrame) -> list[int]:
    return frame.group_by("year_month", maintain_order=True).len().get_column("len").to_list()


def _fit_xgb_ranker(
    *,
    xgb: Any,
    params: dict[str, Any],
    train_df: pl.DataFrame,
    features: Sequence[str],
    config: PortfolioBoostingExactEmaVariantsConfig,
    seed: int,
) -> Any:
    train_rank_df = train_df.sort(["year_month", "ticker"])
    dtrain = xgb.DMatrix(_matrix(train_rank_df, features), label=_target(train_rank_df))
    dtrain.set_group(_group_sizes(train_rank_df))
    train_params = {**XGB_RANK_BASE_PARAMS, **{k: v for k, v in params.items() if k != "num_boost_round"}}
    train_params["objective"] = config.ranker_objective
    train_params["seed"] = int(seed)
    return xgb.train(
        params=train_params,
        dtrain=dtrain,
        num_boost_round=int(params.get("num_boost_round", 180)),
        verbose_eval=False,
    )


def _predict_xgb_ranker(xgb: Any, model: Any, frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    return np.asarray(model.predict(xgb.DMatrix(_matrix(frame, features))), dtype=float).reshape(-1)


def _selected_monthly(scored: pl.DataFrame, score_col: str, top_k: int) -> pl.DataFrame:
    return (
        scored.with_columns(pl.col(score_col).rank(method="ordinal", descending=True).over("year_month").alias("_rank"))
        .filter(pl.col("_rank") <= int(top_k))
        .group_by("year_month")
        .agg(
            pl.mean("future_return").alias("portfolio_return"),
            pl.mean("benchmark_future_return").alias("benchmark_return"),
            pl.mean("future_excess_return").alias("active_return"),
        )
        .sort("year_month")
    )


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + np.nan_to_num(returns, nan=0.0))
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / np.maximum(peak, 1e-12) - 1.0))


def _objective_metric(
    scored: pl.DataFrame,
    score_col: str,
    top_k: int,
    config: PortfolioBoostingExactEmaVariantsConfig,
) -> float:
    monthly = _selected_monthly(scored, score_col, top_k)
    if monthly.is_empty():
        return -999.0
    active = monthly.get_column("active_return").to_numpy().astype(float)
    return float(
        np.nanmean(active)
        - config.robust_vol_penalty * np.nanstd(active)
        + config.robust_dd_penalty * _max_drawdown(active)
    )


def _score_frame(frame: pl.DataFrame, scores: np.ndarray, score_col: str) -> pl.DataFrame:
    return frame.select(
        [
            "ticker",
            "year_month",
            "decision_month",
            "decision_asof_date",
            "holding_month",
            "future_return",
            "benchmark_future_return",
            "future_excess_return",
            "future_excess_rank_target",
            "legacy_selected",
        ]
    ).with_columns(
        pl.Series(score_col, scores, dtype=pl.Float64),
        (pl.col("future_excess_return") > 0.0).cast(pl.Int8).alias("target_label"),
    )


def _tune_xgb_rank_fold(
    *,
    xgb: Any,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    features: Sequence[str],
    config: PortfolioBoostingExactEmaVariantsConfig,
    seed: int,
    fold_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed, n_startup_trials=config.startup_trials))
    study.enqueue_trial(_xgb_base_trial_params())

    def objective(trial: optuna.Trial) -> float:
        params = _xgb_sample_params(trial)
        model = _fit_xgb_ranker(xgb=xgb, params=params, train_df=train_df, features=features, config=config, seed=seed)
        train_scores = _predict_xgb_ranker(xgb, model, train_df, features)
        val_scores = _predict_xgb_ranker(xgb, model, val_df, features)
        train_metric = _objective_metric(_score_frame(train_df, train_scores, "_score"), "_score", config.objective_top_k, config)
        val_metric = _objective_metric(_score_frame(val_df, val_scores, "_score"), "_score", config.objective_top_k, config)
        trial.set_user_attr("train_metric", train_metric)
        trial.set_user_attr("val_metric", val_metric)
        return float(val_metric - config.robust_gap_penalty * abs(train_metric - val_metric))

    study.optimize(objective, n_trials=config.n_trials)
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {
            "fold": fold_label,
            "variant": "boost_rank_pairwise",
            "trial_number": trial.number,
            "objective": trial.value,
            "train_metric": trial.user_attrs.get("train_metric"),
            "val_metric": trial.user_attrs.get("val_metric"),
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        rows.append(row)
    complete = [trial for trial in study.trials if trial.value is not None]
    if not complete:
        raise ValueError(f"No complete rank trial for {fold_label}.")
    return dict(max(complete, key=lambda trial: float(trial.value)).params), rows


def _tune_mlcraft_robust_fold(
    *,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    features: Sequence[str],
    config: PortfolioBoostingExactEmaVariantsConfig,
    seed: int,
    fold_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    search_space = to_mlcraft_search_space(RANK_SEARCH_SPACE)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed, n_startup_trials=config.startup_trials))
    study.enqueue_trial(_base_trial_params(search_space))

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, search_space)
        model = _fit_mlcraft_regressor(params=params, X=_matrix(train_df, features), y=_target(train_df), seed=seed)
        train_scores = _predict(model, _matrix(train_df, features))
        val_scores = _predict(model, _matrix(val_df, features))
        train_metric = _objective_metric(_score_frame(train_df, train_scores, "_score"), "_score", config.objective_top_k, config)
        val_metric = _objective_metric(_score_frame(val_df, val_scores, "_score"), "_score", config.objective_top_k, config)
        trial.set_user_attr("train_metric", train_metric)
        trial.set_user_attr("val_metric", val_metric)
        return float(val_metric - config.robust_gap_penalty * abs(train_metric - val_metric))

    study.optimize(objective, n_trials=config.n_trials)
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {
            "fold": fold_label,
            "variant": "boost_rank_robust_active",
            "trial_number": trial.number,
            "objective": trial.value,
            "train_metric": trial.user_attrs.get("train_metric"),
            "val_metric": trial.user_attrs.get("val_metric"),
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        rows.append(row)
    complete = [trial for trial in study.trials if trial.value is not None]
    if not complete:
        raise ValueError(f"No complete robust trial for {fold_label}.")
    return dict(max(complete, key=lambda trial: float(trial.value)).params), rows


def _binary_top_target(frame: pl.DataFrame, positive_quantile: float) -> np.ndarray:
    target = (
        frame.with_columns(
            (
                pl.col("future_excess_return").rank(method="average").over("year_month")
                / pl.len().over("year_month")
                >= float(positive_quantile)
            )
            .cast(pl.Int8)
            .alias("_target_top")
        )
        .get_column("_target_top")
        .to_numpy()
    )
    return target.astype(np.int8)


def _return_target(frame: pl.DataFrame, clip: float) -> np.ndarray:
    return np.clip(frame.get_column("future_excess_return").to_numpy(), -clip, clip).astype(np.float32)


def _risk_target(frame: pl.DataFrame, clip: float) -> np.ndarray:
    excess = frame.get_column("future_excess_return").to_numpy().astype(float)
    return np.clip(np.maximum(-excess, 0.0), 0.0, clip).astype(np.float32)


def _rank_pct_columns(frame: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    exprs = []
    for col in cols:
        exprs.append((pl.col(col).rank(method="average").over("year_month") / pl.len().over("year_month")).alias(f"{col}_rank_pct"))
    return frame.with_columns(exprs)


def _two_head_predictions(
    *,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    features: Sequence[str],
    config: PortfolioBoostingExactEmaVariantsConfig,
    seed: int,
) -> pl.DataFrame:
    X_train = _matrix(train_df, features)
    X_test = _matrix(test_df, features)
    classifier = _fit_mlcraft_classifier(
        params=CLASSIFIER_BASE_PARAMS,
        X=X_train,
        y=_binary_top_target(train_df, config.positive_quantile),
        seed=seed,
    )
    return_model = _fit_mlcraft_regressor(
        params=RANK_BASE_PARAMS,
        X=X_train,
        y=_return_target(train_df, config.return_clip),
        seed=seed,
    )
    risk_model = _fit_mlcraft_regressor(
        params=RANK_BASE_PARAMS,
        X=X_train,
        y=_risk_target(train_df, config.return_clip),
        seed=seed,
    )
    frame = _score_frame(test_df, np.zeros(test_df.height), "boost_two_head_return_risk").with_columns(
        pl.Series("boost_two_head_proba", _predict(classifier, X_test), dtype=pl.Float64),
        pl.Series("boost_two_head_expected_excess", _predict(return_model, X_test), dtype=pl.Float64),
        pl.Series("boost_two_head_predicted_downside", _predict(risk_model, X_test), dtype=pl.Float64),
    )
    frame = _rank_pct_columns(
        frame,
        ["boost_two_head_proba", "boost_two_head_expected_excess", "boost_two_head_predicted_downside"],
    )
    return frame.with_columns(
        (
            0.50 * pl.col("boost_two_head_proba_rank_pct")
            + 0.35 * pl.col("boost_two_head_expected_excess_rank_pct")
            - 0.15 * pl.col("boost_two_head_predicted_downside_rank_pct")
        ).alias("boost_two_head_return_risk")
    )


def _write_report(
    run_dir: Path,
    *,
    comparison_metrics: pl.DataFrame,
    prediction_metrics: pl.DataFrame,
    config: PortfolioBoostingExactEmaVariantsConfig,
) -> None:
    lines = [
        "# Portfolio boosting exact EMA variants",
        "",
        "But: tester trois pistes pure boosting sur les memes variables EMA exactes.",
        "",
        "- `boost_rank_pairwise` : XGBoost ranking `rank:pairwise` groupe par mois.",
        "- `boost_rank_robust_active` : regression mlcraft du rang futur avec objectif Optuna actif penalise risque.",
        "- `boost_two_head_return_risk` : trois modeles mlcraft, proba top10 + rendement attendu - downside predit.",
        "",
        "Contraintes: pas de score EMA brut en sortie, pas de blend momentum, pas d'objectif Legacy.",
        f"Objectif validation principal: top `{config.objective_top_k}`.",
        f"Trials par fold: `{config.n_trials}`.",
        "",
        "## Backtest",
        "",
        "| modele | total return | CAGR | Sharpe | max DD | vol mensuelle | mois positifs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_metrics.sort("Total Return", descending=True).to_dicts():
        lines.append(
            f"| `{row['model']}` | {row['Total Return'] * 100:.1f}% | {row['CAGR'] * 100:.1f}% | "
            f"{row['Sharpe Ratio']:.2f} | {row['Max Drawdown'] * 100:.1f}% | "
            f"{row['Monthly Volatility'] * 100:.1f}% | {row['Positive Periods %'] * 100:.1f}% |"
        )
    lines.extend(["", "## Top-K Prediction", "", "| modele | top N | metrique | valeur |", "|---|---:|---|---:|"])
    for row in prediction_metrics.to_dicts():
        value = row["value"]
        formatted = f"{value * 100:.2f}%" if row["metric"] in {"avg_monthly_future_excess_return", "hit_rate_gt0"} else f"{value:.4f}"
        lines.append(f"| `{row['model']}` | {row['top_n']} | `{row['metric']}` | {formatted} |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: PortfolioBoostingExactEmaVariantsConfig) -> Path:
    run_dir = config.output_dir / f"portfolio_boosting_exact_ema_variants_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    xgb = load_xgboost()
    base_config = PortfolioBoostingExactEmaRankConfig(
        max_windows=config.max_windows,
        min_train_months=config.min_train_months,
        val_months=config.val_months,
        test_months=config.test_months,
        step_months=config.step_months,
        top_n_values=config.top_n_values,
        seed=config.seed,
    )
    frame, features = _load_exact_ema_rank_frame(base_config)
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
    trial_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for position, window in enumerate(windows, start=1):
        fold = f"fold_{window.fold_index:03d}"
        train_df = filter_by_months(frame, window.train_months)
        val_df = filter_by_months(frame, window.val_months)
        test_df = filter_by_months(frame, window.test_months)
        if train_df.is_empty() or val_df.is_empty() or test_df.is_empty():
            continue
        seed = config.seed + position
        rank_params, rank_rows = _tune_xgb_rank_fold(
            xgb=xgb,
            train_df=train_df,
            val_df=val_df,
            features=features,
            config=config,
            seed=seed,
            fold_label=fold,
        )
        robust_params, robust_rows = _tune_mlcraft_robust_fold(
            train_df=train_df,
            val_df=val_df,
            features=features,
            config=config,
            seed=seed + 10_000,
            fold_label=fold,
        )
        trial_rows.extend(rank_rows)
        trial_rows.extend(robust_rows)
        fit_df = pl.concat([train_df, val_df], how="vertical")
        rank_model = _fit_xgb_ranker(xgb=xgb, params=rank_params, train_df=fit_df, features=features, config=config, seed=seed)
        rank_scores = _predict_xgb_ranker(xgb, rank_model, test_df, features)
        robust_model = _fit_mlcraft_regressor(
            params=robust_params,
            X=_matrix(fit_df, features),
            y=_target(fit_df),
            seed=seed + 10_000,
        )
        robust_scores = _predict(robust_model, _matrix(test_df, features))
        two_head = _two_head_predictions(
            train_df=fit_df,
            test_df=test_df,
            features=features,
            config=config,
            seed=seed + 20_000,
        ).select(
            [
                "ticker",
                "year_month",
                "boost_two_head_proba",
                "boost_two_head_expected_excess",
                "boost_two_head_predicted_downside",
                "boost_two_head_return_risk",
            ]
        )
        scored = (
            _score_frame(test_df, rank_scores, "boost_rank_pairwise")
            .with_columns(
                pl.Series("boost_rank_robust_active", robust_scores, dtype=pl.Float64),
                pl.lit(fold).alias("fold"),
            )
            .join(two_head, on=["ticker", "year_month"], how="left")
        )
        prediction_frames.append(scored)
        fold_rows.append(
            {
                "fold": fold,
                "test_start": str(min(window.test_months)),
                "test_end": str(max(window.test_months)),
                "rank_pairwise_metric": _objective_metric(scored, "boost_rank_pairwise", config.objective_top_k, config),
                "rank_robust_metric": _objective_metric(scored, "boost_rank_robust_active", config.objective_top_k, config),
                "two_head_metric": _objective_metric(scored, "boost_two_head_return_risk", config.objective_top_k, config),
            }
        )
        print(
            f"{fold}: test={window.test_months[0]} "
            f"pair={fold_rows[-1]['rank_pairwise_metric']:.4f} "
            f"robust={fold_rows[-1]['rank_robust_metric']:.4f} "
            f"two_head={fold_rows[-1]['two_head_metric']:.4f}",
            flush=True,
        )

    predictions = pl.concat(prediction_frames, how="vertical")
    predictions.write_parquet(run_dir / "predictions.parquet")
    pl.DataFrame(trial_rows).write_csv(run_dir / "optuna_trials.csv")
    pl.DataFrame(fold_rows).write_csv(run_dir / "fold_metrics.csv")
    _write_warm_start_candidates(run_dir, [row for row in trial_rows if row.get("variant") == "boost_rank_robust_active"], limit=30)
    (run_dir / "xgb_rank_warm_start_candidates.json").write_text(
        json.dumps(
            {
                "candidates": sorted(
                    [row for row in trial_rows if row.get("variant") == "boost_rank_pairwise" and row.get("objective") is not None],
                    key=lambda row: float(row["objective"]),
                    reverse=True,
                )[:30]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    score_cols = ["boost_rank_pairwise", "boost_rank_robust_active", "boost_two_head_return_risk"]
    scenarios = []
    for score_col in score_cols:
        for top_n in config.top_n_values:
            scenarios.append(
                _run_model_scenario(
                    predictions,
                    score_col,
                    f"{score_col}_top_{top_n}",
                    top_n,
                    config.risk_free_rate,
                )
            )
    monthly_returns = pl.concat([scenario["monthly_returns"] for scenario in scenarios], how="vertical")
    selections = pl.concat([scenario["selections"] for scenario in scenarios], how="diagonal_relaxed")
    months_out = (
        predictions.select(pl.col("holding_month").alias("year_month"))
        .unique()
        .sort("year_month")
        .get_column("year_month")
        .to_list()
    )
    comparison_inputs = {
        scenario["name"]: scenario["monthly_returns"].select(
            "year_month",
            pl.col("portfolio_return").alias("monthly_return"),
            pl.col("n_positions").alias("n"),
        )
        for scenario in scenarios
    }
    comparison_inputs["SPY"] = build_spy_curve(predictions)
    comparison_inputs.update(load_legacy_curves(config.legacy_monthly_returns, months_out))
    comparison = compare_backtest_curves(
        comparison_inputs,
        output_path=run_dir / "comparison.html",
        title="Portfolio boosting exact EMA variants vs Legacy",
        risk_free_rate=config.risk_free_rate,
    )
    prediction_metrics = pl.concat(
        [
            _prediction_metrics(predictions, score_col, config.top_n_values).with_columns(pl.lit(score_col).alias("model"))
            for score_col in score_cols
        ],
        how="vertical",
    ).select(["model", "top_n", "metric", "value"])
    monthly_returns.write_parquet(run_dir / "monthly_returns.parquet")
    selections.write_parquet(run_dir / "selections.parquet")
    prediction_metrics.write_csv(run_dir / "prediction_metrics.csv")
    comparison.metrics.write_csv(run_dir / "comparison_metrics.csv")
    comparison.annual_returns.write_csv(run_dir / "annual_returns.csv")
    comparison.correlation_matrix.write_csv(run_dir / "correlation_matrix.csv")
    comparison.worst_periods.write_csv(run_dir / "worst_periods.csv")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                "feature_count": len(features),
                "features": features,
                "score_semantics": "All final scores are produced only from boosting model predictions.",
                "variant_notes": {
                    "boost_rank_pairwise": "Native XGBoost rank:pairwise because mlcraft does not expose ranking TaskSpec yet.",
                    "boost_rank_robust_active": "mlcraft XGBoost regression on future monthly rank with robust active-return validation objective.",
                    "boost_two_head_return_risk": "mlcraft classifier + return regressor + downside regressor; final score combines only boosted predictions.",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(run_dir, comparison_metrics=comparison.metrics, prediction_metrics=prediction_metrics, config=config)
    print(f"RUN_DIR={run_dir}")
    print(comparison.metrics.sort("Total Return", descending=True).head(30))
    return run_dir


def _parse_args() -> PortfolioBoostingExactEmaVariantsConfig:
    parser = argparse.ArgumentParser(description="Run pure-boosting exact EMA model variants.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--legacy-monthly-returns", type=Path, default=DEFAULT_LEGACY_MONTHLY_RETURNS)
    parser.add_argument("--n-trials", type=int, default=2)
    parser.add_argument("--startup-trials", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=999)
    parser.add_argument("--min-train-months", type=int, default=168)
    parser.add_argument("--val-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--objective-top-k", type=int, default=10)
    parser.add_argument("--top-n", type=int, nargs="*", default=[5, 7, 10, 20, 30, 50])
    parser.add_argument("--ranker-objective", choices=["rank:pairwise", "rank:ndcg"], default="rank:pairwise")
    parser.add_argument("--return-clip", type=float, default=0.30)
    parser.add_argument("--positive-quantile", type=float, default=0.90)
    parser.add_argument("--robust-vol-penalty", type=float, default=0.35)
    parser.add_argument("--robust-dd-penalty", type=float, default=0.25)
    parser.add_argument("--robust-gap-penalty", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return PortfolioBoostingExactEmaVariantsConfig(
        output_dir=args.output_dir,
        legacy_monthly_returns=args.legacy_monthly_returns,
        n_trials=args.n_trials,
        startup_trials=args.startup_trials,
        max_windows=args.max_windows,
        min_train_months=args.min_train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        step_months=args.step_months,
        objective_top_k=args.objective_top_k,
        top_n_values=tuple(args.top_n),
        ranker_objective=args.ranker_objective,
        return_clip=args.return_clip,
        positive_quantile=args.positive_quantile,
        robust_vol_penalty=args.robust_vol_penalty,
        robust_dd_penalty=args.robust_dd_penalty,
        robust_gap_penalty=args.robust_gap_penalty,
        seed=args.seed,
    )


if __name__ == "__main__":
    run(_parse_args())
