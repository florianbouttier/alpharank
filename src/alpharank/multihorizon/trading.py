from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import date, datetime

import numpy as np
import polars as pl


def _turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    names = set(previous) | set(current)
    return 0.5 * sum(abs(current.get(name, 0.0) - previous.get(name, 0.0)) for name in names)


def build_monthly_top_n_returns(
    predictions: pl.DataFrame,
    *,
    top_n: int,
    transaction_cost_bps: float,
) -> pl.DataFrame:
    """Equal-weight, monthly-rebalanced portfolio from out-of-sample scores."""

    rows: list[dict] = []
    previous_weights: dict[str, float] = {}
    usable = predictions.filter(
        pl.col("future_return_1m").is_not_null()
        & pl.col("benchmark_future_return_1m").is_not_null()
    )
    for month_frame in usable.partition_by("decision_month", maintain_order=True):
        selected = month_frame.sort("score", descending=True).head(top_n)
        tickers = selected["ticker"].to_list()
        if not tickers:
            continue
        weight = 1.0 / len(tickers)
        weights = {ticker: weight for ticker in tickers}
        turnover = _turnover(previous_weights, weights) if previous_weights else 1.0
        gross_return = float(selected["future_return_1m"].mean())
        benchmark_return = float(selected["benchmark_future_return_1m"][0])
        cost = turnover * float(transaction_cost_bps) / 10_000.0
        rows.append(
            {
                "decision_month": selected["decision_month"][0],
                "holding_month": selected["decision_month"][0].replace(day=1),
                "top_n": int(top_n),
                "n_positions": len(tickers),
                "gross_return": gross_return,
                "net_return": gross_return - cost,
                "benchmark_return": benchmark_return,
                "turnover": turnover,
                "transaction_cost": cost,
            }
        )
        previous_weights = weights
    result = pl.DataFrame(rows).sort("decision_month")
    return result.with_columns(pl.col("decision_month").dt.offset_by("1mo").alias("holding_month"))


def performance_statistics(returns: np.ndarray) -> dict[str, float]:
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {
            "total_return": float("nan"),
            "cagr": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "positive_month_rate": float("nan"),
        }
    curve = np.cumprod(1.0 + clean)
    total = float(curve[-1] - 1.0)
    years = clean.size / 12.0
    cagr = float(curve[-1] ** (1.0 / years) - 1.0) if curve[-1] > 0 and years > 0 else -1.0
    volatility = float(np.std(clean, ddof=1) * math.sqrt(12.0)) if clean.size > 1 else 0.0
    sharpe = (
        float(np.mean(clean) / np.std(clean, ddof=1) * math.sqrt(12.0))
        if clean.size > 1 and np.std(clean, ddof=1) > 0
        else float("nan")
    )
    running_peak = np.maximum.accumulate(curve)
    max_drawdown = float(np.min(curve / running_peak - 1.0))
    return {
        "total_return": total,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "positive_month_rate": float(np.mean(clean > 0.0)),
    }


def legacy_report_statistics(
    returns: np.ndarray,
    *,
    holding_months: Iterable[date | datetime | np.datetime64],
    risk_free_rate: float = 0.02,
) -> dict[str, float | int]:
    """Return performance metrics using the canonical Legacy report convention.

    The Legacy HTML reports define Sharpe as ``(CAGR - risk_free_rate) /
    annualized_volatility``. The worst year is restricted to full calendar
    years so partial boundary years cannot make strategies look artificially
    better or worse.
    """

    values = np.asarray(returns, dtype=float)
    months = np.asarray(list(holding_months))
    if values.size != months.size:
        raise ValueError("returns and holding_months must have the same length")
    valid = np.isfinite(values) & ~np.isnat(months.astype("datetime64[ns]"))
    values = values[valid]
    months = months[valid].astype("datetime64[M]")
    base = performance_statistics(values)
    volatility = float(base["annualized_volatility"])
    canonical_sharpe = (
        float((float(base["cagr"]) - risk_free_rate) / volatility)
        if volatility > 0.0
        else float("nan")
    )

    annual_returns: list[tuple[int, float]] = []
    if values.size:
        years = months.astype("datetime64[Y]").astype(int) + 1970
        month_numbers = (
            months.astype(int) - months.astype("datetime64[Y]").astype(int) * 12
        ) + 1
        for year in np.unique(years):
            in_year = years == year
            if set(month_numbers[in_year].tolist()) != set(range(1, 13)):
                continue
            annual_returns.append(
                (int(year), float(np.prod(1.0 + values[in_year]) - 1.0))
            )
    if annual_returns:
        worst_year, worst_year_return = min(
            annual_returns,
            key=lambda item: item[1],
        )
    else:
        worst_year, worst_year_return = -1, float("nan")

    return {
        **base,
        "sharpe": canonical_sharpe,
        "risk_free_rate": float(risk_free_rate),
        "worst_full_calendar_year": worst_year,
        "worst_full_calendar_year_return": worst_year_return,
        "full_calendar_years": len(annual_returns),
    }


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
