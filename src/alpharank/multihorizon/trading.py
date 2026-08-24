from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import polars as pl

from alpharank.portfolio.adapters.boosting import boosting_predictions_to_holdings
from alpharank.portfolio.performance import (
    legacy_report_statistics,
    performance_statistics,
)
from alpharank.portfolio.simulation import simulate_weighted_portfolio


def build_monthly_top_n_returns(
    predictions: pl.DataFrame,
    *,
    top_n: int,
    transaction_cost_bps: float,
) -> pl.DataFrame:
    """Equal-weight, monthly-rebalanced portfolio from out-of-sample scores."""
    strategy = f"top_{int(top_n)}"
    holdings = boosting_predictions_to_holdings(
        predictions,
        strategy=strategy,
        top_n=int(top_n),
    )
    monthly = simulate_weighted_portfolio(
        holdings.select(
            "strategy",
            "decision_month",
            "holding_month",
            "ticker",
            "target_weight",
            "realized_return",
            "benchmark_return",
        ),
        transaction_cost_bps=transaction_cost_bps,
        causal_timing_policy="legacy_month_only",
    )
    return monthly.select(
        "decision_month",
        "holding_month",
        pl.lit(int(top_n)).alias("top_n"),
        "n_positions",
        "gross_return",
        "net_return",
        "benchmark_return",
        "turnover",
        "transaction_cost",
    )


def summarize_monthly_backtest(monthly: pl.DataFrame) -> dict[str, float | int | object]:
    net = monthly["net_return"].to_numpy()
    gross = monthly["gross_return"].to_numpy()
    benchmark = monthly["benchmark_return"].to_numpy()
    active = net - benchmark
    gross_metrics = performance_statistics(gross)
    net_metrics = performance_statistics(net)
    benchmark_metrics = performance_statistics(benchmark)
    tracking_error = float(np.std(active, ddof=1) * math.sqrt(12.0)) if len(active) > 1 else float("nan")
    information_ratio = (
        float(np.mean(active) / np.std(active, ddof=1) * math.sqrt(12.0))
        if len(active) > 1 and np.std(active, ddof=1) > 0
        else float("nan")
    )
    return {
        "start_decision_month": monthly["decision_month"].min(),
        "end_decision_month": monthly["decision_month"].max(),
        "months": monthly.height,
        "gross_total_return": gross_metrics["total_return"],
        "gross_cagr": gross_metrics["cagr"],
        "net_total_return": net_metrics["total_return"],
        "net_cagr": net_metrics["cagr"],
        "net_annualized_volatility": net_metrics["annualized_volatility"],
        "net_sharpe": net_metrics["sharpe"],
        "net_max_drawdown": net_metrics["max_drawdown"],
        "net_positive_month_rate": net_metrics["positive_month_rate"],
        "benchmark_total_return": benchmark_metrics["total_return"],
        "benchmark_cagr": benchmark_metrics["cagr"],
        "benchmark_sharpe": benchmark_metrics["sharpe"],
        "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
        "annualized_tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "average_monthly_turnover": float(monthly["turnover"].mean()),
        "total_transaction_cost": float(monthly["transaction_cost"].sum()),
    }


def evaluate_trading_predictions(
    predictions: pl.DataFrame,
    *,
    top_n_values: Iterable[int],
    transaction_cost_bps: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    monthly_parts: list[pl.DataFrame] = []
    summaries: list[dict] = []
    for top_n in top_n_values:
        monthly = build_monthly_top_n_returns(
            predictions,
            top_n=int(top_n),
            transaction_cost_bps=transaction_cost_bps,
        )
        monthly_parts.append(monthly)
        summaries.append({"top_n": int(top_n), **summarize_monthly_backtest(monthly)})
    return pl.concat(monthly_parts), pl.DataFrame(summaries)
