from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class PortfolioMaturitySplit:
    """Predictions split without using a selected stock's future availability."""

    completed_predictions: pl.DataFrame
    score_only_predictions: pl.DataFrame
    manifest: dict[str, Any]


def _date_text(value: object | None) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


def split_completed_portfolio_months(
    predictions: pl.DataFrame,
    *,
    decision_month_column: str = "decision_month",
    benchmark_return_column: str = "benchmark_future_return_1m",
    benchmark_tolerance: float = 1e-12,
) -> PortfolioMaturitySplit:
    """Separate realized portfolio months from a recent score-only tail.

    A decision month enters the economic replay only when the one-month
    benchmark return is finite and consistent on every prediction row. A month
    with no benchmark return remains available as a live score-only month. A
    partially populated or inconsistent benchmark month fails closed.

    Stock-return availability is deliberately not inspected here: selection
    must happen before that check, and the common simulator remains responsible
    for rejecting a missing return on a selected stock.
    """

    if benchmark_tolerance < 0.0:
        raise ValueError("benchmark_tolerance must be non-negative.")
    required = {decision_month_column, benchmark_return_column}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Missing portfolio maturity columns: {missing}")

    if predictions.is_empty():
        return PortfolioMaturitySplit(
            completed_predictions=predictions.clone(),
            score_only_predictions=predictions.clone(),
            manifest={
                "policy_id": "complete_one_month_benchmark_v1",
                "decision_months_total": 0,
                "completed_decision_months": 0,
                "score_only_decision_months": 0,
                "first_completed_decision_month": None,
                "last_completed_decision_month": None,
                "score_only_months": [],
                "stock_return_rule": (
                    "selected stock returns are checked after ranking and still fail closed"
                ),
            },
        )

    prepared = predictions.with_columns(
        pl.col(benchmark_return_column)
        .cast(pl.Float64, strict=False)
        .alias("_portfolio_maturity_benchmark")
    )
    month_status = (
        prepared.group_by(decision_month_column)
        .agg(
            pl.len().alias("row_count"),
            pl.col("_portfolio_maturity_benchmark")
            .is_finite()
            .fill_null(False)
            .sum()
            .alias("finite_benchmark_count"),
            pl.col("_portfolio_maturity_benchmark")
            .filter(pl.col("_portfolio_maturity_benchmark").is_finite().fill_null(False))
            .min()
            .alias("minimum_benchmark_return"),
            pl.col("_portfolio_maturity_benchmark")
            .filter(pl.col("_portfolio_maturity_benchmark").is_finite().fill_null(False))
            .max()
            .alias("maximum_benchmark_return"),
        )
        .sort(decision_month_column)
    )

    partial = month_status.filter(
        (pl.col("finite_benchmark_count") > 0)
        & (pl.col("finite_benchmark_count") < pl.col("row_count"))
    )
    if partial.height:
        months = partial[decision_month_column].cast(pl.Utf8).to_list()
        raise ValueError(
            "Partially observed one-month benchmark return for decision months: "
            f"{months}"
        )

    inconsistent = month_status.filter(
        (pl.col("finite_benchmark_count") == pl.col("row_count"))
        & (
            (
                pl.col("maximum_benchmark_return")
                - pl.col("minimum_benchmark_return")
            ).abs()
            > benchmark_tolerance
        )
    )
    if inconsistent.height:
        months = inconsistent[decision_month_column].cast(pl.Utf8).to_list()
        raise ValueError(
            "Inconsistent one-month benchmark return for decision months: "
            f"{months}"
        )

    completed_months = month_status.filter(
        pl.col("finite_benchmark_count") == pl.col("row_count")
    )[decision_month_column].to_list()
    score_only_months = month_status.filter(
        pl.col("finite_benchmark_count") == 0
    )[decision_month_column].to_list()
    completed = predictions.filter(pl.col(decision_month_column).is_in(completed_months))
    score_only = predictions.filter(pl.col(decision_month_column).is_in(score_only_months))

    return PortfolioMaturitySplit(
        completed_predictions=completed,
        score_only_predictions=score_only,
        manifest={
            "policy_id": "complete_one_month_benchmark_v1",
            "decision_months_total": month_status.height,
            "completed_decision_months": len(completed_months),
            "score_only_decision_months": len(score_only_months),
            "first_completed_decision_month": _date_text(
                completed_months[0] if completed_months else None
            ),
            "last_completed_decision_month": _date_text(
                completed_months[-1] if completed_months else None
            ),
            "score_only_months": [_date_text(value) for value in score_only_months],
            "benchmark_rule": (
                "every prediction row in a decision month has one finite and "
                "consistent one-month benchmark return"
            ),
            "stock_return_rule": (
                "selected stock returns are checked after ranking and still fail closed"
            ),
        },
    )
