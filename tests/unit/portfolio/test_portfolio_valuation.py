from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.portfolio.allocation import drifted_weights_after_returns
from alpharank.portfolio.simulation import simulate_weighted_portfolio
from alpharank.portfolio.valuation import ValuationWindow, value_frozen_holdings

ENTRY = date(2026, 9, 30)
START = date(2026, 10, 30)
END = date(2026, 11, 20)


def _holdings() -> pl.DataFrame:
    return pl.DataFrame(
        {"strategy": ["Frozen", "Frozen"], "ticker": ["A", "B"], "target_weight": [0.5, 0.5]}
    )


def _prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["A"] * 3 + ["B"] * 3 + ["SPY.US"] * 3,
            "date": [ENTRY, START, END] * 3,
            "adjusted_close": [100.0, 200.0, 220.0, 100.0, 100.0, 90.0, 100.0, 105.0, 110.0],
        }
    )


def test_subperiod_uses_drifted_weights_not_a_silent_rebalance() -> None:
    result = value_frozen_holdings(
        _holdings(), _prices(), entry_date=ENTRY, windows=[ValuationWindow("November", START, END)]
    ).row(0, named=True)
    drifted = drifted_weights_after_returns({"A": 0.5, "B": 0.5}, {"A": 1.0, "B": 0.0})

    assert result["total_return"] == pytest.approx(drifted["A"] * 0.1 - drifted["B"] * 0.1)
    assert result["benchmark_return"] == pytest.approx(110.0 / 105.0 - 1.0)
    assert result["total_return"] != pytest.approx(0.0)


def test_chained_windows_reproduce_the_full_buy_and_hold_return() -> None:
    result = value_frozen_holdings(
        _holdings(),
        _prices(),
        entry_date=ENTRY,
        windows=[
            ValuationWindow("October", ENTRY, START),
            ValuationWindow("November", START, END),
            ValuationWindow("All", ENTRY, END),
        ],
    )
    returns = dict(result.select("window", "total_return").iter_rows())

    assert (1 + returns["October"]) * (1 + returns["November"]) - 1 == pytest.approx(
        returns["All"], abs=1e-12
    )
    assert returns["All"] == pytest.approx(0.55)


def test_future_prices_cannot_change_an_earlier_valuation() -> None:
    windows = [ValuationWindow("October", ENTRY, START)]
    altered = _prices().with_columns(
        pl.when(pl.col("date") > START)
        .then(pl.lit(1e6))
        .otherwise(pl.col("adjusted_close"))
        .alias("adjusted_close")
    )

    assert value_frozen_holdings(_holdings(), _prices(), entry_date=ENTRY, windows=windows).equals(
        value_frozen_holdings(_holdings(), altered, entry_date=ENTRY, windows=windows)
    )


def test_missing_selected_price_fails_without_dropping_a_holding() -> None:
    prices = _prices().filter(~((pl.col("ticker") == "A") & (pl.col("date") == END)))

    with pytest.raises(ValueError, match="calendar"):
        value_frozen_holdings(
            _holdings(), prices, entry_date=ENTRY, windows=[ValuationWindow("All", ENTRY, END)]
        )


@pytest.mark.parametrize("bad_price", [None, float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_price_fails(bad_price: float | None) -> None:
    prices = _prices().with_columns(
        pl.when((pl.col("ticker") == "A") & (pl.col("date") == END))
        .then(pl.lit(bad_price, dtype=pl.Float64))
        .otherwise(pl.col("adjusted_close"))
        .alias("adjusted_close")
    )
    with pytest.raises(ValueError, match="price"):
        value_frozen_holdings(
            _holdings(), prices, entry_date=ENTRY, windows=[ValuationWindow("All", ENTRY, END)]
        )


@pytest.mark.parametrize("weights", [[0.4, 0.5], [float("nan"), 0.5], [-0.1, 1.1]])
def test_invalid_target_weights_fail(weights: list[float]) -> None:
    holdings = _holdings().with_columns(pl.Series("target_weight", weights))
    with pytest.raises(ValueError, match="weight"):
        value_frozen_holdings(
            holdings, _prices(), entry_date=ENTRY, windows=[ValuationWindow("All", ENTRY, END)]
        )


def test_duplicate_price_fails() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        value_frozen_holdings(
            _holdings(),
            pl.concat([_prices(), _prices().head(1)]),
            entry_date=ENTRY,
            windows=[ValuationWindow("All", ENTRY, END)],
        )


def test_completed_holding_month_matches_shared_simulation_gross_return() -> None:
    holdings = _holdings().with_columns(
        pl.lit(date(2026, 9, 1)).alias("decision_month"),
        pl.lit(date(2026, 10, 1)).alias("holding_month"),
        pl.Series("realized_return", [1.0, 0.0]),
        pl.lit(0.05).alias("benchmark_return"),
    )
    # The explicit historical escape hatch reproduces a frozen month, not a new causal signal.
    simulated = simulate_weighted_portfolio(
        holdings, transaction_cost_bps=10.0, causal_timing_policy="legacy_month_only"
    )
    valued = value_frozen_holdings(
        holdings, _prices(), entry_date=ENTRY, windows=[ValuationWindow("October", ENTRY, START)]
    )

    assert valued["total_return"][0] == pytest.approx(simulated["gross_return"][0], abs=1e-12)
    assert valued["benchmark_return"][0] == pytest.approx(simulated["benchmark_return"][0])


@pytest.mark.parametrize(
    "windows",
    [
        [],
        [ValuationWindow("Same", ENTRY, START), ValuationWindow("Same", START, END)],
        [ValuationWindow("Before entry", date(2026, 9, 29), END)],
        [ValuationWindow("Empty", START, START)],
        [ValuationWindow("Absent endpoint", date(2026, 10, 29), END)],
    ],
)
def test_invalid_windows_fail(windows: list[ValuationWindow]) -> None:
    with pytest.raises(ValueError):
        value_frozen_holdings(_holdings(), _prices(), entry_date=ENTRY, windows=windows)


def test_missing_benchmark_fails() -> None:
    with pytest.raises(ValueError, match="calendar"):
        value_frozen_holdings(
            _holdings(),
            _prices().filter(pl.col("ticker") != "SPY.US"),
            entry_date=ENTRY,
            windows=[ValuationWindow("All", ENTRY, END)],
        )
