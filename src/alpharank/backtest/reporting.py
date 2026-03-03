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
    if not auc_train or not auc_val:
        return None

    rounds = np.arange(1, len(auc_train) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, auc_train, label="Train AUC", color="#1f77b4")
    plt.plot(rounds, auc_val, label="Validation AUC", color="#ff7f0e")
    plt.title(f"{fold_label} Learning Curve (AUC)")
    plt.xlabel("Boosting round")
    plt.ylabel("AUC")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_lift_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    bins: int,
    path: Path,
    fold_label: str,
) -> Path | None:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.size == 0:
        return None

    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]

    overall_rate = np.mean(y_sorted)
    if overall_rate <= 0:
        overall_rate = 1e-12

    quantiles = np.linspace(0.1, 1.0, bins)
    lift_values: List[float] = []

    for q in quantiles:
        k = max(1, int(np.ceil(q * y_sorted.size)))
        captured_rate = np.mean(y_sorted[:k])
        lift_values.append(float(captured_rate / overall_rate))

    plt.figure(figsize=(9, 5))
    plt.plot(quantiles * 100.0, lift_values, marker="o", color="#2ca02c")
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.title(f"{fold_label} Lift Curve")
    plt.xlabel("Top fraction of scored universe (%)")
    plt.ylabel("Lift")
    plt.grid(alpha=0.25)
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
    plt.figure(figsize=(10, 5))
    plt.plot(x, score, marker="o", color="#d62728")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title("Objective Score by Fold")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(score_path, dpi=150)
    plt.close()
    paths["score_by_fold"] = score_path

    return paths


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


def write_html_report(
    title: str,
    output_path: Path,
    kpis: pl.DataFrame,
    fold_metrics: pl.DataFrame,
    best_params: pl.DataFrame,
    global_images: Dict[str, Path],
    fold_assets: List[Dict[str, Any]],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def img_tag(path: Path | None) -> str:
        if path is None:
            return ""
        try:
            rel = path.relative_to(output_path.parent).as_posix()
        except Exception:
            rel = path.name
        return f"<div><img src='{html.escape(rel)}' style='max-width: 1100px; width: 100%;'/></div>"

    sections: List[str] = [
        f"<h1>{html.escape(title)}</h1>",
        "<h2>Backtest KPIs</h2>",
        _polars_to_html_table(kpis, precision=6),
        "<h2>Fold Metrics</h2>",
        _polars_to_html_table(fold_metrics, precision=6),
        "<h2>Best Hyperparameters by Fold</h2>",
        _polars_to_html_table(best_params, precision=6),
    ]

    if global_images:
        sections.append("<h2>Global Visualizations</h2>")
        for _, path in sorted(global_images.items()):
            sections.append(img_tag(path))

    if fold_assets:
        sections.append("<h2>Per-Fold Analysis</h2>")
        for idx, assets in enumerate(fold_assets, start=1):
            label = str(assets.get("__label__", f"Fold {idx}"))
            sections.append(f"<h3>{html.escape(label)}</h3>")
            for key, path in sorted(assets.items()):
                if key == "__label__":
                    continue
                sections.append(img_tag(path))

    html_content = (
        "<html><head><meta charset='utf-8'/>"
        "<title>Backtest Report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:24px auto;padding:0 20px;}"
        "table{border-collapse:collapse;margin-bottom:18px;}th{background:#f3f3f3;}"
        "h1,h2,h3{margin-top:26px;}</style>"
        "</head><body>"
        + "\n".join(sections)
        + "</body></html>"
    )

    output_path.write_text(html_content, encoding="utf-8")
    return output_path
