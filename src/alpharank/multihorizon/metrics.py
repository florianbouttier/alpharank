from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, ndcg_score, roc_auc_score


def _safe_metric(function, *args) -> float:
    try:
        value = float(function(*args))
        return value if math.isfinite(value) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def score_predictions(
    predictions: pl.DataFrame,
    *,
    method: str,
    horizon: int,
    top_n_values: Iterable[int],
) -> tuple[dict[str, float], pl.DataFrame]:
    y = predictions.get_column(f"future_excess_return_{horizon}m").to_numpy()
    score = predictions.get_column("score").to_numpy()
    rank_target = predictions.get_column(f"future_excess_rank_{horizon}m").to_numpy()
    selected = predictions.get_column("legacy_selected").to_numpy().astype(bool)
    monthly_ic: list[float] = []
    monthly_ndcg: list[float] = []
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
    metrics: dict[str, float] = {
        "spearman_ic": float(np.nanmean(monthly_ic)),
        "ndcg": float(np.nanmean(monthly_ndcg)),
    }
    if method in {"classification", "teacher"}:
        target = (
            selected.astype(int)
            if method == "teacher"
            else (rank_target >= 0.90).astype(int)
        )
        metrics.update(
            roc_auc=_safe_metric(roc_auc_score, target, score),
            average_precision=_safe_metric(average_precision_score, target, score),
            brier=_safe_metric(brier_score_loss, target, score),
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
