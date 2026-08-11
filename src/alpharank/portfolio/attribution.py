from __future__ import annotations

import math

import numpy as np
import polars as pl

from alpharank.portfolio.contracts import validate_holdings, validate_monthly_returns


ATTRIBUTION_COLUMNS = (
    "strategy",
    "decision_month",
    "holding_month",
    "component",
    "component_type",
    "simple_return_contribution",
    "log_return_contribution",
    "monthly_net_return",
)


def _log_allocation_factor(monthly_return: float) -> float:
    if monthly_return <= -1.0:
        raise ValueError("Log attribution requires every monthly return to exceed -100%.")
    return math.log1p(monthly_return) / monthly_return if abs(monthly_return) > 1e-15 else 1.0


def portfolio_return_attribution(
    holdings: pl.DataFrame,
    monthly_returns: pl.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> pl.DataFrame:
    """Allocate exact monthly net log return to securities and trading costs."""

    validate_holdings(holdings)
    validate_monthly_returns(monthly_returns)
    monthly = monthly_returns.select(
        "strategy",
        "decision_month",
        "holding_month",
        "net_return",
        "transaction_cost",
    )
    rows: list[dict[str, object]] = []
    matched_months: set[tuple[str, object, object]] = set()
    for month in holdings.sort(
        ["strategy", "decision_month", "ticker"]
    ).partition_by(
        ["strategy", "decision_month", "holding_month"],
        maintain_order=True,
    ):
        strategy = str(month["strategy"][0])
        decision_month = month["decision_month"][0]
        holding_month = month["holding_month"][0]
        key = (strategy, decision_month, holding_month)
        result = monthly.filter(
            (pl.col("strategy") == strategy)
            & (pl.col("decision_month") == decision_month)
            & (pl.col("holding_month") == holding_month)
        )
        if result.height != 1:
            raise ValueError(f"Expected one monthly return row for {key}; got {result.height}.")
        matched_months.add(key)
        net_return = float(result["net_return"][0])
        transaction_cost = float(result["transaction_cost"][0])
        target_weights = month["target_weight"].to_numpy().astype(float)
        realized = month["realized_return"].to_numpy().astype(float)
        available = np.isfinite(realized)
        if not np.any(available):
            raise ValueError(f"No realized return is available for {key}.")
        effective_weights = np.zeros_like(target_weights)
        effective_weights[available] = (
            target_weights[available] / target_weights[available].sum()
        )
        factor = _log_allocation_factor(net_return)
        simple_sum = 0.0
        for index, ticker in enumerate(month["ticker"].to_list()):
            if not available[index]:
                continue
            simple = float(effective_weights[index] * realized[index])
            simple_sum += simple
            rows.append(
                {
                    "strategy": strategy,
                    "decision_month": decision_month,
                    "holding_month": holding_month,
                    "component": str(ticker),
                    "component_type": "security",
                    "target_weight": float(target_weights[index]),
                    "effective_weight": float(effective_weights[index]),
                    "realized_return": float(realized[index]),
                    "simple_return_contribution": simple,
                    "log_return_contribution": simple * factor,
                    "monthly_net_return": net_return,
                }
            )
        if transaction_cost:
            simple_sum -= transaction_cost
            rows.append(
                {
                    "strategy": strategy,
                    "decision_month": decision_month,
                    "holding_month": holding_month,
                    "component": "Transaction costs",
                    "component_type": "cost",
                    "target_weight": None,
                    "effective_weight": None,
                    "realized_return": None,
                    "simple_return_contribution": -transaction_cost,
                    "log_return_contribution": -transaction_cost * factor,
                    "monthly_net_return": net_return,
                }
            )
        if abs(simple_sum - net_return) > tolerance:
            raise ValueError(
                f"Attribution does not reproduce monthly net return for {key}: "
                f"error={simple_sum - net_return}."
            )
    expected_months = {
        (str(row[0]), row[1], row[2])
        for row in monthly.select(
            "strategy", "decision_month", "holding_month"
        ).iter_rows()
    }
    if matched_months != expected_months:
        missing = sorted(expected_months - matched_months)
        raise ValueError(f"Attribution is missing monthly holdings: {missing[:3]}")
    result = pl.DataFrame(rows).sort(
        ["strategy", "decision_month", "component_type", "component"]
    )
    reconciliation = result.group_by(
        "strategy", "decision_month", "holding_month"
    ).agg(
        pl.col("log_return_contribution").sum().alias("attributed_log_return"),
        pl.col("monthly_net_return").first().alias("monthly_net_return"),
    )
    maximum_error = reconciliation.select(
        (
            pl.col("attributed_log_return")
            - pl.col("monthly_net_return").log1p()
        )
        .abs()
        .max()
    ).item()
    if maximum_error is None or maximum_error > tolerance:
        raise ValueError(f"Log-return attribution error={maximum_error}.")
    return result


def reference_return_attribution(
    monthly_returns: pl.DataFrame,
    *,
    component: str,
) -> pl.DataFrame:
    """Represent a one-component reference series in the attribution contract."""

    validate_monthly_returns(monthly_returns)
    return (
        monthly_returns.select(
            "strategy",
            "decision_month",
            "holding_month",
            pl.lit(component).alias("component"),
            pl.lit("reference").alias("component_type"),
            pl.lit(1.0).alias("target_weight"),
            pl.lit(1.0).alias("effective_weight"),
            pl.col("net_return").alias("realized_return"),
            pl.col("net_return").alias("simple_return_contribution"),
            pl.col("net_return").log1p().alias("log_return_contribution"),
            pl.col("net_return").alias("monthly_net_return"),
        )
        .sort(["strategy", "decision_month"])
    )
