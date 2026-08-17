from __future__ import annotations

from datetime import date

import math
import polars as pl
import pytest

from alpharank.portfolio.attribution import portfolio_return_attribution
from alpharank.portfolio.simulation import simulate_weighted_portfolio


def test_attribution_reconciles_securities_costs_and_cagr() -> None:
    holdings = pl.DataFrame(
        {
            "strategy": ["alpha"] * 4,
            "decision_month": [date(2020, 1, 1)] * 2 + [date(2020, 2, 1)] * 2,
            "holding_month": [date(2020, 2, 1)] * 2 + [date(2020, 3, 1)] * 2,
            "ticker": ["A", "B", "A", "C"],
            "target_weight": [0.5] * 4,
            "realized_return": [0.10, -0.02, 0.04, 0.08],
            "benchmark_return": [0.01] * 4,
        }
    )
    monthly = simulate_weighted_portfolio(
        holdings,
        transaction_cost_bps=10.0,
        causal_timing_policy="legacy_month_only",
    )

    attribution = portfolio_return_attribution(holdings, monthly)
    reconciled = attribution.group_by("holding_month").agg(
        pl.col("simple_return_contribution").sum().alias("simple"),
        pl.col("log_return_contribution").sum().alias("log"),
        pl.col("monthly_net_return").first().alias("net"),
    )

    assert reconciled["simple"].to_list() == pytest.approx(
        reconciled["net"].to_list()
    )
    assert reconciled["log"].to_list() == pytest.approx(
        [math.log1p(value) for value in reconciled["net"]]
    )
    annualized_log = 6.0 * attribution["log_return_contribution"].sum()
    expected_cagr = math.prod(1.0 + value for value in monthly["net_return"]) ** 6 - 1
    assert math.expm1(annualized_log) == pytest.approx(expected_cagr)
    assert attribution.filter(pl.col("component_type") == "cost").height == 2


def test_attribution_uses_effective_weights_when_return_is_missing() -> None:
    holdings = pl.DataFrame(
        {
            "strategy": ["legacy", "legacy"],
            "decision_month": [date(2020, 1, 1)] * 2,
            "holding_month": [date(2020, 2, 1)] * 2,
            "ticker": ["A", "B"],
            "target_weight": [0.75, 0.25],
            "realized_return": [0.10, None],
            "benchmark_return": [0.01, 0.01],
        }
    )
    monthly = simulate_weighted_portfolio(
        holdings,
        missing_return_policy="renormalize_available",
        causal_timing_policy="legacy_month_only",
    )

    attribution = portfolio_return_attribution(holdings, monthly)

    assert attribution.height == 1
    assert attribution["component"][0] == "A"
    assert attribution["effective_weight"][0] == pytest.approx(1.0)
    assert attribution["simple_return_contribution"][0] == pytest.approx(0.10)
