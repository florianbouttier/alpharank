from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    r2_score,
    roc_auc_score,
)


def _safe_metric(function, *args) -> float:
    try:
        value = float(function(*args))
        return value if math.isfinite(value) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def _nanmean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _expected_calibration_error(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, len(target))
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (
            probability <= upper if upper == 1.0 else probability < upper
        )
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(float(target[mask].mean()) - float(probability[mask].mean()))
    return error * len(target) / total


def score_predictions(
    predictions: pl.DataFrame,
    *,
    method: str,
    horizon: int,
    top_n_values: Iterable[int],
) -> tuple[dict[str, float], pl.DataFrame]:
    y = predictions.get_column(f"future_excess_return_{horizon}m").to_numpy()
    score = predictions.get_column("score").to_numpy()
    probability = (
        predictions.get_column("calibrated_probability").to_numpy()
        if "calibrated_probability" in predictions.columns
        else score
    )
    rank_target = predictions.get_column(f"future_excess_rank_{horizon}m").to_numpy()
    selected = predictions.get_column("legacy_selected").to_numpy().astype(bool)
    monthly_ic: list[float] = []
    monthly_ndcg: list[float] = []
    monthly_ndcg_at_5: list[float] = []
    monthly_ndcg_at_10: list[float] = []
    monthly_ndcg_at_20: list[float] = []
    monthly_ndcg_at_10_no_signal: list[float] = []
    for month_frame in predictions.partition_by("decision_month", maintain_order=True):
        month_score = month_frame["score"].to_numpy()
        month_y = month_frame[f"future_excess_return_{horizon}m"].to_numpy()
        month_rank = month_frame[f"future_excess_rank_{horizon}m"].to_numpy()
        monthly_ic.append(
            _safe_metric(lambda a, b: spearmanr(a, b).statistic, month_score, month_y)
        )
        monthly_ndcg.append(
            _safe_metric(lambda a, b: ndcg_score(a[None, :], b[None, :]), month_rank, month_score)
        )
        monthly_ndcg_at_5.append(
            _safe_metric(
                lambda a, b: ndcg_score(a[None, :], b[None, :], k=5),
                month_rank,
                month_score,
            )
        )
        monthly_ndcg_at_10.append(
            _safe_metric(
                lambda a, b: ndcg_score(a[None, :], b[None, :], k=10),
                month_rank,
                month_score,
            )
        )
        monthly_ndcg_at_10_no_signal.append(
            _safe_metric(
                lambda a, b: ndcg_score(a[None, :], b[None, :], k=10),
                month_rank,
                np.zeros_like(month_score),
            )
        )
        monthly_ndcg_at_20.append(
            _safe_metric(
                lambda a, b: ndcg_score(a[None, :], b[None, :], k=20),
                month_rank,
                month_score,
            )
        )
    ndcg_at_10 = _nanmean(monthly_ndcg_at_10)
    ndcg_at_10_no_signal = _nanmean(monthly_ndcg_at_10_no_signal)
    metrics: dict[str, float] = {
        "spearman_ic": _nanmean(monthly_ic),
        "ndcg": _nanmean(monthly_ndcg),
        "ndcg_at_5": _nanmean(monthly_ndcg_at_5),
        "ndcg_at_10": ndcg_at_10,
        "ndcg_at_10_no_signal": ndcg_at_10_no_signal,
        "ndcg_at_10_lift": ndcg_at_10 - ndcg_at_10_no_signal,
        "ndcg_at_20": _nanmean(monthly_ndcg_at_20),
    }
    if method in {"classification", "teacher"}:
        target = (
            selected.astype(int)
            if method == "teacher"
            else (rank_target >= 0.90).astype(int)
        )
        pr_auc = _safe_metric(average_precision_score, target, score)
        positive_rate = float(target.mean())
        metrics.update(
            roc_auc=_safe_metric(roc_auc_score, target, score),
            pr_auc_average_precision=pr_auc,
            pr_auc_lift_vs_prevalence=pr_auc / positive_rate if positive_rate > 0 else float("nan"),
            brier=_safe_metric(brier_score_loss, target, probability),
            log_loss=_safe_metric(
                log_loss,
                target,
                np.clip(probability, 1e-6, 1 - 1e-6),
            ),
            expected_calibration_error=_expected_calibration_error(target, probability),
            positive_rate=positive_rate,
        )
    if method == "regression":
        target_std = float(np.std(y))
        rmse = _safe_metric(lambda a, b: mean_squared_error(a, b) ** 0.5, y, score)
        metrics.update(
            rmse=rmse,
            normalized_rmse=rmse / target_std if target_std > 0 else float("nan"),
            mae=_safe_metric(mean_absolute_error, y, score),
            r2=_safe_metric(r2_score, y, score),
            target_mean=float(np.mean(y)),
            target_std=target_std,
        )
    portfolio_rows: list[dict] = []
    for month_frame in predictions.partition_by("decision_month", maintain_order=True):
        ordered = month_frame.sort("score", descending=True)
        legacy_names = set(month_frame.filter(pl.col("legacy_selected") == 1).get_column("ticker").to_list())
        for top_n in top_n_values:
            picked = ordered.head(top_n)
            names = set(picked.get_column("ticker").to_list())
            # Canonical project definition: common names / Legacy basket size.
            overlap_denominator = max(1, len(legacy_names))
            portfolio_rows.append(
                {
                    "decision_month": month_frame.get_column("decision_month")[0],
                    "top_n": int(top_n),
                    "future_excess_return": float(
                        picked.get_column(f"future_excess_return_{horizon}m").mean()
                    ),
                    "realized_one_month_excess": float(
                        picked.get_column("future_excess_return_1m").mean()
                    ),
                    "legacy_overlap": len(names & legacy_names) / overlap_denominator,
                    "legacy_jaccard": len(names & legacy_names) / max(1, len(names | legacy_names)),
                }
            )
    portfolio = pl.DataFrame(portfolio_rows)
    for top_n in top_n_values:
        subset = portfolio.filter(pl.col("top_n") == top_n)
        metrics[f"top{top_n}_horizon_excess"] = float(subset["future_excess_return"].mean())
        metrics[f"top{top_n}_one_month_excess"] = float(subset["realized_one_month_excess"].mean())
        metrics[f"top{top_n}_legacy_overlap"] = float(subset["legacy_overlap"].mean())
    return metrics, portfolio
