from __future__ import annotations

import numpy as np
import polars as pl

from alpharank.portfolio.allocation import portfolio_turnover
from alpharank.portfolio.contracts import (
    empty_monthly_returns,
    validate_holdings,
    validate_monthly_returns,
)


def simulate_weighted_portfolio(
    holdings: pl.DataFrame,
    *,
    transaction_cost_bps: float = 0.0,
    missing_return_policy: str = "raise",
    validate: bool = True,
) -> pl.DataFrame:
    """Simulate monthly long-only holdings through one canonical return engine.

    Missing selected returns fail closed by default. Callers replaying the
    historical Legacy convention may request ``renormalize_available``
    explicitly; that policy excludes missing returns and scales the remaining
    weights to one for performance while leaving target weights unchanged for
    turnover.
    """

    if transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be non-negative.")
    if missing_return_policy not in {"renormalize_available", "raise"}:
        raise ValueError(f"Unsupported missing_return_policy={missing_return_policy!r}.")
    if holdings.is_empty():
        return empty_monthly_returns()
    if validate:
        validate_holdings(holdings)

    rows: list[dict[str, object]] = []
    previous_by_strategy: dict[str, dict[str, float]] = {}
    ordered = holdings.sort(["strategy", "decision_month", "ticker"])
    for month in ordered.partition_by(
        ["strategy", "decision_month", "holding_month"],
        maintain_order=True,
    ):
        strategy = str(month["strategy"][0])
        target_weights = month["target_weight"].to_numpy().astype(float)
        tickers = month["ticker"].to_list()
        current = dict(zip(tickers, target_weights, strict=True))
        turnover = portfolio_turnover(previous_by_strategy.get(strategy, {}), current)

        realized = month["realized_return"].to_numpy().astype(float)
        available = np.isfinite(realized)
        if not np.all(available) and missing_return_policy == "raise":
            raise ValueError(
                f"Missing realized return for strategy={strategy}, "
                f"decision_month={month['decision_month'][0]}."
            )
        if not np.any(available):
            raise ValueError(
                f"No realized return is available for strategy={strategy}, "
                f"decision_month={month['decision_month'][0]}."
            )
        performance_weights = target_weights[available]
        performance_weights = performance_weights / performance_weights.sum()
        gross_return = float(np.dot(performance_weights, realized[available]))

        benchmark_values = month["benchmark_return"].to_numpy().astype(float)
        finite_benchmark = benchmark_values[np.isfinite(benchmark_values)]
        if finite_benchmark.size == 0:
            raise ValueError("No finite benchmark return is available for a portfolio month.")
        benchmark_return = float(finite_benchmark[0])
        if not np.allclose(finite_benchmark, benchmark_return, rtol=0.0, atol=1e-12):
            raise ValueError("Benchmark return is inconsistent inside a portfolio month.")

        transaction_cost = turnover * float(transaction_cost_bps) / 10_000.0
        net_return = gross_return - transaction_cost
        active_return = net_return - benchmark_return
        relative_return = (
            (1.0 + net_return) / (1.0 + benchmark_return) - 1.0
            if abs(1.0 + benchmark_return) > 1e-12
            else float("nan")
        )
        sector_count = 0
        maximum_sector_weight = 0.0
        if "sector" in month.columns:
            sectors = month.with_columns(
                pl.col("sector").fill_null("Unknown").cast(pl.Utf8)
            ).group_by("sector").agg(pl.col("target_weight").sum().alias("weight"))
            sector_count = sectors.height
            maximum_sector_weight = float(sectors["weight"].max())
        rows.append(
            {
                "strategy": strategy,
                "decision_month": month["decision_month"][0],
                "holding_month": month["holding_month"][0],
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "active_return": active_return,
                "relative_return": relative_return,
                "n_positions": month.height,
                "maximum_position_weight": float(np.max(target_weights)),
                "maximum_sector_weight": maximum_sector_weight,
                "sector_count": sector_count,
            }
        )
        previous_by_strategy[strategy] = current

    result = pl.DataFrame(rows).sort(["strategy", "decision_month"])
    if validate:
        validate_monthly_returns(result)
    return result
