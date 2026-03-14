from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_learning_curve(
    evals_result: Dict[str, Dict[str, List[float]]],
    path: Path,
    fold_label: str,
) -> Path | None:
    if not evals_result:
        return None

    train = evals_result.get("validation_0", {})
    val = evals_result.get("validation_1", {})

    auc_train = train.get("auc")
    auc_val = val.get("auc")
    logloss_train = train.get("logloss")
    logloss_val = val.get("logloss")

    has_auc = bool(auc_train) and bool(auc_val)
    has_logloss = bool(logloss_train) and bool(logloss_val)
    if not has_auc and not has_logloss:
        return None

    n_rows = int(has_auc) + int(has_logloss)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4.8 * n_rows), squeeze=False)
    axes_list = axes.flatten()
    plot_idx = 0

    if has_auc:
        rounds = np.arange(1, len(auc_train) + 1)
        ax = axes_list[plot_idx]
        ax.plot(rounds, auc_train, label="Train AUC", color="#1f77b4")
        ax.plot(rounds, auc_val, label="Validation AUC", color="#ff7f0e")
        ax.set_title(f"{fold_label} Learning Curve (AUC)")
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("AUC")
        ax.grid(alpha=0.25)
        ax.legend()
        plot_idx += 1

    if has_logloss:
        rounds = np.arange(1, len(logloss_train) + 1)
        ax = axes_list[plot_idx]
        ax.plot(rounds, logloss_train, label="Train Logloss", color="#2ca02c")
        ax.plot(rounds, logloss_val, label="Validation Logloss", color="#d62728")
        ax.set_title(f"{fold_label} Learning Curve (Logloss)")
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("Logloss")
        ax.grid(alpha=0.25)
        ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _bucket_frequency_frame(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_buckets: int,
) -> pl.DataFrame:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.size == 0:
        return pl.DataFrame({"bucket": [], "predicted_rate": [], "realized_rate": [], "n_obs": []})

    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    indices = np.array_split(np.arange(y_sorted.size), max(1, int(n_buckets)))

    rows: List[Dict[str, float]] = []
    for idx, bucket_idx in enumerate(indices, start=1):
        if bucket_idx.size == 0:
            continue
        rows.append(
            {
                "bucket": float(idx),
                "predicted_rate": float(np.mean(s_sorted[bucket_idx])),
                "realized_rate": float(np.mean(y_sorted[bucket_idx])),
                "n_obs": float(bucket_idx.size),
            }
        )

    return pl.DataFrame(rows)


def save_bucket_frequency_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_buckets: int,
    path: Path,
    fold_label: str,
    split_label: str,
) -> Path | None:
    frame = _bucket_frequency_frame(y_true=y_true, y_score=y_score, n_buckets=n_buckets)
    if frame.is_empty():
        return None

    x = frame.get_column("bucket").to_numpy()
    pred = frame.get_column("predicted_rate").to_numpy()
    real = frame.get_column("realized_rate").to_numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(x, pred, marker="o", color="#1f77b4", label="Predicted frequency")
    plt.plot(x, real, marker="o", color="#d62728", label="Realized frequency")
    plt.title(f"{fold_label} {split_label} Bucket Frequency (sorted, {int(n_buckets)} buckets)")
    plt.xlabel("Bucket rank (1 = highest prediction)")
    plt.ylabel("Positive frequency")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_lift_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_buckets: int,
    path: Path,
    fold_label: str,
    split_label: str,
) -> Path | None:
    frame = _bucket_frequency_frame(y_true=y_true, y_score=y_score, n_buckets=n_buckets)
    if frame.is_empty():
        return None

    x = frame.get_column("bucket").to_numpy()
    pred = frame.get_column("predicted_rate").to_numpy()
    real = frame.get_column("realized_rate").to_numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(x, pred, marker="o", color="#1f77b4", linewidth=2, label="Mean predicted")
    plt.plot(x, real, marker="o", color="#d62728", linewidth=2, label="Mean realized")
    plt.title(f"{fold_label} {split_label} Lift Curve ({int(n_buckets)} ranked buckets)")
    plt.xlabel("Bucket rank (1 = highest prediction)")
    plt.ylabel("Positive frequency")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_auc_score_overview(fold_metrics: pl.DataFrame, out_dir: Path) -> Dict[str, Path]:
    _ensure_dir(out_dir)
    paths: Dict[str, Path] = {}

    if fold_metrics.is_empty():
        return paths

    sorted_df = fold_metrics.sort("fold")

    x = sorted_df.get_column("fold").to_numpy()
    train_auc = sorted_df.get_column("train_auc").to_numpy()
    val_auc = sorted_df.get_column("val_auc").to_numpy()
    test_auc = sorted_df.get_column("test_auc").to_numpy()

    auc_path = out_dir / "auc_by_fold.png"
    plt.figure(figsize=(10, 6))
    plt.plot(x, train_auc, marker="o", label="Train AUC", color="#1f77b4")
    plt.plot(x, val_auc, marker="o", label="Validation AUC", color="#ff7f0e")
    plt.plot(x, test_auc, marker="o", label="Test AUC", color="#2ca02c")
    plt.xlabel("Fold")
    plt.ylabel("AUC")
    plt.title("AUC by Fold")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(auc_path, dpi=150)
    plt.close()
    paths["auc_by_fold"] = auc_path

    score_path = out_dir / "score_by_fold.png"
    score = sorted_df.get_column("objective_score").to_numpy()
    test_penalty = np.maximum(0.0, test_auc - score)
    plt.figure(figsize=(10, 5))
    plt.plot(x, test_auc, marker="o", color="#2ca02c", label="Raw test AUC")
    plt.plot(x, score, marker="o", color="#d62728", label="Penalized test score")
    plt.bar(x, test_penalty, alpha=0.25, color="#9467bd", label="Penalty amount")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title("Score by Fold (AUC_test penalized by train-test gap)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(score_path, dpi=150)
    plt.close()
    paths["score_by_fold"] = score_path

    return paths


def _safe_implied_penalty_weight(penalty_amount: pl.Expr, gap: pl.Expr) -> pl.Expr:
    return (
        pl.when(gap.abs() > 1e-12)
        .then(penalty_amount / gap)
        .otherwise(None)
    )


def build_optimization_focus_table(fold_metrics: pl.DataFrame) -> pl.DataFrame:
    required_cols = {
        "fold",
        "train_auc",
        "val_auc",
        "test_auc",
        "objective_score_val",
        "objective_score",
    }
    if fold_metrics.is_empty() or not required_cols.issubset(set(fold_metrics.columns)):
        return pl.DataFrame(
            schema={
                "fold": pl.Int64,
                "train_auc": pl.Float64,
                "val_auc": pl.Float64,
                "test_auc": pl.Float64,
                "train_val_gap_abs": pl.Float64,
                "train_test_gap_abs": pl.Float64,
                "val_penalty": pl.Float64,
                "test_penalty": pl.Float64,
                "objective_score_val": pl.Float64,
                "objective_score": pl.Float64,
                "implied_lambda_val": pl.Float64,
                "implied_lambda_test": pl.Float64,
            }
        )

    return (
        fold_metrics.sort("fold")
        .with_columns(
            [
                (pl.col("train_auc") - pl.col("val_auc")).abs().alias("train_val_gap_abs"),
                (pl.col("train_auc") - pl.col("test_auc")).abs().alias("train_test_gap_abs"),
                (pl.col("val_auc") - pl.col("objective_score_val")).alias("val_penalty"),
                (pl.col("test_auc") - pl.col("objective_score")).alias("test_penalty"),
            ]
        )
        .with_columns(
            [
                _safe_implied_penalty_weight(pl.col("val_penalty"), pl.col("train_val_gap_abs")).alias(
                    "implied_lambda_val"
                ),
                _safe_implied_penalty_weight(pl.col("test_penalty"), pl.col("train_test_gap_abs")).alias(
                    "implied_lambda_test"
                ),
            ]
        )
        .select(
            [
                "fold",
                "train_auc",
                "val_auc",
                "test_auc",
                "train_val_gap_abs",
                "train_test_gap_abs",
                "val_penalty",
                "test_penalty",
                "objective_score_val",
                "objective_score",
                "implied_lambda_val",
                "implied_lambda_test",
            ]
        )
    )


def _numeric_columns(df: pl.DataFrame, exclude: Iterable[str]) -> List[str]:
    excluded = set(exclude)
    numeric_dtypes = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
    return [
        col
        for col, dtype in df.schema.items()
        if col not in excluded and dtype in numeric_dtypes
    ]


def save_hyperparams_overview(best_params_df: pl.DataFrame, out_dir: Path) -> Dict[str, Path]:
    _ensure_dir(out_dir)
    paths: Dict[str, Path] = {}

    if best_params_df.is_empty():
        return paths

    best_params_df = best_params_df.sort("fold")
    folds = best_params_df.get_column("fold").to_numpy()
    param_cols = _numeric_columns(best_params_df, exclude=["fold"])

    for param in param_cols:
        path = out_dir / f"hyperparam_{param}.png"
        values = best_params_df.get_column(param).to_numpy()

        plt.figure(figsize=(9, 4))
        plt.plot(folds, values, marker="o", color="#9467bd")
        plt.title(f"Best {param} by Fold")
        plt.xlabel("Fold")
        plt.ylabel(param)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        paths[param] = path

    return paths


def save_optuna_trials_curve(
    trials_rows: List[Dict[str, Any]],
    path: Path,
    fold_label: str,
) -> Path | None:
    if not trials_rows:
        return None

    x = [row.get("trial_number", i) for i, row in enumerate(trials_rows)]
    y = [row.get("objective") for row in trials_rows]
    y = [float(v) if v is not None else np.nan for v in y]

    if not np.isfinite(np.asarray(y)).any():
        return None

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker="o", color="#17becf")
    plt.title(f"{fold_label} Optuna Objective by Trial")
    plt.xlabel("Trial")
    plt.ylabel("Objective score")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_best_params_bar(
    best_params: Dict[str, Any],
    path: Path,
    fold_label: str,
) -> Path | None:
    if not best_params:
        return None

    numeric_items = [(k, v) for k, v in best_params.items() if isinstance(v, (int, float))]
    if not numeric_items:
        return None

    labels = [k for k, _ in numeric_items]
    values = [float(v) for _, v in numeric_items]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color="#8c564b")
    plt.title(f"{fold_label} Best Hyperparameters")
    plt.xlabel("Value")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_optuna_visualizations(
    study,
    out_dir: Path,
    fold_label: str,
) -> Dict[str, Path]:
    _ensure_dir(out_dir)
    paths: Dict[str, Path] = {}
    if study is None:
        return paths

    try:
        from optuna import visualization as ov
    except Exception:
        return paths

    plotters = {
        "optimization_history": "plot_optimization_history",
        "slice": "plot_slice",
        "param_importances": "plot_param_importances",
        "timeline": "plot_timeline",
        "terminator_improvement": "plot_terminator_improvement",
    }

    for name, fn_name in plotters.items():
        fn = getattr(ov, fn_name, None)
        if fn is None:
            continue
        try:
            fig = fn(study)
            out = out_dir / f"{fold_label}_optuna_{name}.html"
            fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
            paths[f"optuna_{name}"] = out
        except Exception:
            continue

    return paths


def save_backtest_vs_sp500_plots(monthly_returns: pl.DataFrame, out_dir: Path) -> Dict[str, Path]:
    _ensure_dir(out_dir)
    paths: Dict[str, Path] = {}
    if monthly_returns.is_empty():
        return paths

    sorted_df = monthly_returns.sort("year_month")
    x = np.arange(sorted_df.height)
    p = sorted_df.get_column("portfolio_return").to_numpy()
    b = sorted_df.get_column("benchmark_return").to_numpy()
    a = sorted_df.get_column("active_return").to_numpy()

    cum_p = np.cumprod(1.0 + p)
    cum_b = np.cumprod(1.0 + b)

    cumulative_path = out_dir / "backtest_cumulative_portfolio_vs_sp500.png"
    plt.figure(figsize=(11, 6))
    plt.plot(x, cum_p, label="Portfolio", color="#1f77b4")
    plt.plot(x, cum_b, label="SP500", color="#ff7f0e")
    plt.title("Cumulative Performance: Portfolio vs SP500")
    plt.xlabel("Month index")
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cumulative_path, dpi=150)
    plt.close()
    paths["backtest_cumulative"] = cumulative_path

    dd_p = cum_p / np.maximum.accumulate(cum_p) - 1.0
    dd_b = cum_b / np.maximum.accumulate(cum_b) - 1.0

    drawdown_path = out_dir / "backtest_drawdown_portfolio_vs_sp500.png"
    plt.figure(figsize=(11, 6))
    plt.plot(x, dd_p, label="Portfolio", color="#1f77b4")
    plt.plot(x, dd_b, label="SP500", color="#ff7f0e")
    plt.title("Drawdown: Portfolio vs SP500")
    plt.xlabel("Month index")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(drawdown_path, dpi=150)
    plt.close()
    paths["backtest_drawdown"] = drawdown_path

    active_path = out_dir / "backtest_active_return.png"
    plt.figure(figsize=(11, 5))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.plot(x, a, color="#2ca02c")
    plt.title("Active Return vs SP500 (Monthly)")
    plt.xlabel("Month index")
    plt.ylabel("Active return")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(active_path, dpi=150)
    plt.close()
    paths["backtest_active_return"] = active_path

    return paths


def _polars_to_html_table(df: pl.DataFrame, precision: int = 4) -> str:
    headers = "".join(f"<th>{html.escape(col)}</th>" for col in df.columns)

    rows_html: List[str] = []
    for row in df.iter_rows(named=True):
        cells: List[str] = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                cell = f"{val:.{precision}f}"
            else:
                cell = str(val)
            cells.append(f"<td>{html.escape(cell)}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<table border='1' cellspacing='0' cellpadding='6'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )


def build_backtest_fold_focus_table(
    fold_backtest_kpis: pl.DataFrame,
    fold_index: pl.DataFrame,
) -> pl.DataFrame:
    required = {"fold", "strategy"}
    if fold_backtest_kpis.is_empty() or not required.issubset(set(fold_backtest_kpis.columns)):
        return pl.DataFrame(
            schema={
                "fold": pl.Int64,
                "strategy": pl.Utf8,
                "test_month_start": pl.Utf8,
                "test_month_end": pl.Utf8,
                "total_return": pl.Float64,
                "cagr": pl.Float64,
                "avg_monthly_return": pl.Float64,
                "annualized_volatility": pl.Float64,
                "sharpe_ratio": pl.Float64,
                "sortino_ratio": pl.Float64,
                "calmar_ratio": pl.Float64,
                "max_drawdown": pl.Float64,
                "win_rate": pl.Float64,
                "months": pl.Float64,
                "avg_hit_rate": pl.Float64,
                "avg_positions": pl.Float64,
            }
        )

    test_period_cols = [
        col for col in ["fold", "test_month_start", "test_month_end", "test_rows", "status", "skip_reason"] if col in fold_index.columns
    ]
    if test_period_cols:
        fold_focus = fold_backtest_kpis.join(fold_index.select(test_period_cols), on="fold", how="left")
    else:
        fold_focus = fold_backtest_kpis

    strategy_order = (
        pl.when(pl.col("strategy") == "Portfolio").then(0)
        .when(pl.col("strategy") == "Benchmark").then(1)
        .when(pl.col("strategy") == "Active").then(2)
        .otherwise(99)
        .alias("__strategy_order")
    )
    return fold_focus.with_columns(strategy_order).sort(["fold", "__strategy_order"]).drop("__strategy_order")


def write_html_report(
    title: str,
    output_path: Path,
    backtest_kpis: pl.DataFrame,
    split_kpis: pl.DataFrame,
    fold_metrics: pl.DataFrame,
    best_params: pl.DataFrame,
    global_assets: Dict[str, Path],
    fold_assets: List[Dict[str, Any]],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimization_focus = build_optimization_focus_table(fold_metrics)

    def asset_tag(path: Path | None) -> str:
        if path is None:
            return ""
        try:
            rel = path.relative_to(output_path.parent).as_posix()
        except Exception:
            rel = path.name

        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
            return f"<div><img src='{html.escape(rel)}' style='max-width: 1100px; width: 100%;'/></div>"
        if suffix == ".html":
            return (
                f"<div><iframe src='{html.escape(rel)}' "
                "style='width: 100%; height: 620px; border: 1px solid #ddd;'></iframe></div>"
            )
        if suffix == ".pdf":
            return f"<div><a href='{html.escape(rel)}' target='_blank'>Open {html.escape(path.name)}</a></div>"

        return f"<div><a href='{html.escape(rel)}' target='_blank'>{html.escape(path.name)}</a></div>"

    sections: List[str] = [
        f"<h1>{html.escape(title)}</h1>",
        "<h2>Optimization Metric Focus</h2>",
        (
            "<p>Primary objective per fold: penalized AUC. "
            "Validation optimization score = AUC_val - penalty. "
            "Backtest summary score = AUC_test - penalty. "
            "The penalty grows with the absolute train-vs-validation or train-vs-test AUC gap, "
            "so a fold with high raw AUC but poor generalization is pushed down.</p>"
        ),
        _polars_to_html_table(optimization_focus, precision=6),
        "<h2>Model KPIs (Train / Validation / Test)</h2>",
        _polars_to_html_table(split_kpis, precision=6),
        "<h2>Fold Metrics</h2>",
        _polars_to_html_table(fold_metrics, precision=6),
        "<h2>Best Hyperparameters by Fold</h2>",
        _polars_to_html_table(best_params, precision=6),
        "<h2>Backtest KPIs (Portfolio vs SP500)</h2>",
        _polars_to_html_table(backtest_kpis, precision=6),
    ]

    if global_assets:
        sections.append("<h2>Global Visualizations</h2>")
        for key, path in sorted(global_assets.items()):
            sections.append(f"<h4>{html.escape(str(key))}</h4>")
            sections.append(asset_tag(path))

    optuna_assets_by_fold: List[tuple[str, List[tuple[str, Path]]]] = []
    for idx, assets in enumerate(fold_assets, start=1):
        label = str(assets.get("__label__", f"Fold {idx}"))
        optuna_items = [
            (str(key), path)
            for key, path in sorted(assets.items())
            if key != "__label__" and (str(key).startswith("optuna_") or str(key) == "optuna_trials")
        ]
        if optuna_items:
            optuna_assets_by_fold.append((label, optuna_items))

    if optuna_assets_by_fold:
        sections.append("<h2>Optuna Visualizations</h2>")
        for label, items in optuna_assets_by_fold:
            sections.append(f"<h3>{html.escape(label)}</h3>")
            for key, path in items:
                sections.append(f"<h4>{html.escape(key)}</h4>")
                sections.append(asset_tag(path))

    lift_assets_by_fold: List[tuple[str, List[tuple[str, Path]]]] = []
    for idx, assets in enumerate(fold_assets, start=1):
        label = str(assets.get("__label__", f"Fold {idx}"))
        lift_items = [
            (str(key), path)
            for key, path in sorted(assets.items())
            if key != "__label__" and str(key).startswith("lift_")
        ]
        if lift_items:
            lift_assets_by_fold.append((label, lift_items))

    if lift_assets_by_fold:
        sections.append("<h2>Lift Curves</h2>")
        for label, items in lift_assets_by_fold:
            sections.append(f"<h3>{html.escape(label)}</h3>")
            for key, path in items:
                sections.append(f"<h4>{html.escape(key)}</h4>")
                sections.append(asset_tag(path))

    html_content = (
        "<html><head><meta charset='utf-8'/>"
        "<title>Backtest Report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:24px auto;padding:0 20px;}"
        "table{border-collapse:collapse;margin-bottom:18px;}th{background:#f3f3f3;}"
        "h1,h2,h3{margin-top:26px;} iframe{margin-bottom:18px;}</style>"
        "</head><body>"
        + "\n".join(sections)
        + "</body></html>"
    )

    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def write_backtest_audit_report(
    output_path: Path,
    monthly_returns: pl.DataFrame,
    fold_monthly_returns: pl.DataFrame,
    backtest_kpis: pl.DataFrame,
    fold_backtest_kpis: pl.DataFrame,
    selections: pl.DataFrame,
    debug_predictions_long: pl.DataFrame,
    fold_index: pl.DataFrame,
    linked_artifacts: Dict[str, Path],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fold_focus = build_backtest_fold_focus_table(fold_backtest_kpis, fold_index)

    def asset_link(path: Path | None) -> str:
        if path is None:
            return ""
        try:
            rel = path.relative_to(output_path.parent).as_posix()
        except Exception:
            rel = path.name
        return f"<li><a href='{html.escape(rel)}' target='_blank'>{html.escape(path.name)}</a></li>"

    def plotly_div(fig) -> str:
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    monthly_pdf = monthly_returns.to_pandas() if not monthly_returns.is_empty() else None
    selections_pdf = selections.to_pandas() if not selections.is_empty() else None
    debug_pdf = debug_predictions_long.to_pandas() if not debug_predictions_long.is_empty() else None

    plots: List[str] = []
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception:
        px = None
        go = None

    if px is not None and monthly_pdf is not None and not monthly_pdf.empty:
        cumulative_df = monthly_pdf.copy()
        cumulative_df["portfolio_growth"] = (1.0 + cumulative_df["portfolio_return"]).cumprod()
        cumulative_df["benchmark_growth"] = (1.0 + cumulative_df["benchmark_return"]).cumprod()

        cumulative_fig = go.Figure()
        cumulative_fig.add_trace(
            go.Scatter(x=cumulative_df["year_month"], y=cumulative_df["portfolio_growth"], mode="lines", name="Portfolio")
        )
        cumulative_fig.add_trace(
            go.Scatter(x=cumulative_df["year_month"], y=cumulative_df["benchmark_growth"], mode="lines", name="SP500")
        )
        cumulative_fig.update_layout(title="Cumulative Performance", height=500)
        cumulative_fig.update_xaxes(rangeslider_visible=True)
        plots.append(plotly_div(cumulative_fig))

        active_fig = px.bar(
            monthly_pdf,
            x="year_month",
            y="active_return",
            title="Monthly Active Return (holding month)",
            hover_data=["decision_month", "portfolio_return", "benchmark_return", "hit_rate", "n_positions"],
        )
        active_fig.update_xaxes(rangeslider_visible=True)
        active_fig.update_layout(height=500)
        plots.append(plotly_div(active_fig))

    if px is not None and debug_pdf is not None and not debug_pdf.empty:
        scored_fig = px.scatter(
            debug_pdf,
            x="prediction",
            y="future_excess_return",
            color="selected_top_n",
            hover_data=[
                "decision_month",
                "holding_month",
                "ticker",
                "fold",
                "prediction_rank_in_month",
                "future_return",
                "benchmark_future_return",
                "holding_period_complete",
            ],
            title="Prediction vs Future Excess Return (all scored rows)",
        )
        scored_fig.update_layout(height=550)
        plots.append(plotly_div(scored_fig))

    if px is not None and selections_pdf is not None and not selections_pdf.empty:
        bought_fig = px.scatter(
            selections_pdf,
            x="prediction",
            y="future_excess_return",
            color="decision_month",
            hover_data=[
                "decision_month",
                "holding_month",
                "ticker",
                "fold",
                "rank",
                "future_return",
                "benchmark_future_return",
                "future_relative_return",
                "holding_period_complete",
            ],
            title="Bought Positions: prediction vs realized excess return",
        )
        bought_fig.update_layout(height=600)
        plots.append(plotly_div(bought_fig))

        monthly_selection_fig = px.strip(
            selections_pdf,
            x="holding_month",
            y="future_excess_return",
            color="fold",
            hover_data=[
                "decision_month",
                "holding_month",
                "ticker",
                "prediction",
                "rank",
                "future_return",
                "benchmark_future_return",
                "holding_period_complete",
            ],
            title="Bought Positions by Holding Month",
        )
        monthly_selection_fig.update_xaxes(rangeslider_visible=True)
        monthly_selection_fig.update_layout(height=600)
        plots.append(plotly_div(monthly_selection_fig))

    best_months = (
        monthly_returns.sort("active_return", descending=True).head(15)
        if not monthly_returns.is_empty()
        else monthly_returns
    )
    worst_months = (
        monthly_returns.sort("active_return", descending=False).head(15)
        if not monthly_returns.is_empty()
        else monthly_returns
    )
    best_positions = (
        selections.sort("future_excess_return", descending=True).head(30)
        if not selections.is_empty()
        else selections
    )
    worst_positions = (
        selections.sort("future_excess_return", descending=False).head(30)
        if not selections.is_empty()
        else selections
    )

    fold_sections: List[str] = []
    if not fold_monthly_returns.is_empty() and not fold_focus.is_empty():
        for fold_key, frame in fold_monthly_returns.partition_by("fold", as_dict=True).items():
            fold_id = int(fold_key[0]) if isinstance(fold_key, tuple) else int(fold_key)
            fold_kpi_table = fold_focus.filter(pl.col("fold") == fold_id)
            fold_meta = fold_index.filter(pl.col("fold") == fold_id)
            period_label = ""
            if not fold_meta.is_empty():
                meta_row = fold_meta.to_dicts()[0]
                period_label = (
                    f"test={meta_row.get('test_month_start', 'n/a')} -> {meta_row.get('test_month_end', 'n/a')}"
                )
            fold_sections.append(f"<h3>Fold {fold_id:02d} {html.escape(period_label)}</h3>")
            if not fold_meta.is_empty():
                fold_sections.append(_polars_to_html_table(fold_meta, precision=6))
            fold_sections.append(_polars_to_html_table(fold_kpi_table, precision=6))
            fold_sections.append(_polars_to_html_table(frame.sort("holding_month"), precision=6))

    linked = "".join(asset_link(path) for _, path in sorted(linked_artifacts.items()))
    html_content = (
        "<html><head><meta charset='utf-8'/>"
        "<title>Backtest Report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;}"
        "table{border-collapse:collapse;margin-bottom:18px;}th{background:#f3f3f3;}"
        "h1,h2,h3{margin-top:26px;} details{margin:16px 0;} iframe{margin-bottom:18px;}</style>"
        "</head><body>"
        "<h1>Backtest Report</h1>"
        "<p>This HTML is specialized on realized trading behavior, with explicit KPI focus by test fold period.</p>"
        "<h2>Global Backtest KPIs</h2>"
        f"{_polars_to_html_table(backtest_kpis, precision=6)}"
        "<h2>Test Fold KPI Focus</h2>"
        f"{_polars_to_html_table(fold_focus, precision=6)}"
        "<h2>Linked Artifacts</h2>"
        f"<ul>{linked}</ul>"
        "<h2>Fold Coverage</h2>"
        f"{_polars_to_html_table(fold_index, precision=6)}"
        "<h2>Interactive Charts</h2>"
        + "\n".join(plots)
        + "<h2>Fold-by-Fold Test Period Breakdown</h2>"
        + "\n".join(fold_sections)
        + "<h2>Best Months</h2>"
        + _polars_to_html_table(best_months, precision=6)
        + "<h2>Worst Months</h2>"
        + _polars_to_html_table(worst_months, precision=6)
        + "<h2>Best Bought Positions</h2>"
        + _polars_to_html_table(best_positions, precision=6)
        + "<h2>Worst Bought Positions</h2>"
        + _polars_to_html_table(worst_positions, precision=6)
        + "<details><summary>All bought positions</summary>"
        + _polars_to_html_table(selections.sort(["year_month", "rank"]), precision=6)
        + "</details>"
        + "<details><summary>All scored rows</summary>"
        + _polars_to_html_table(debug_predictions_long.sort(["year_month", "prediction_rank_in_month", "ticker"]), precision=6)
        + "</details>"
        + "</body></html>"
    )

    output_path.write_text(html_content, encoding="utf-8")
    return output_path
