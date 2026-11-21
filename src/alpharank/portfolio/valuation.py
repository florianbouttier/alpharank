"""Mark frozen allocations to market without generating signals or rebalance orders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from math import fsum

import numpy as np
import polars as pl

from alpharank.portfolio.allocation import drifted_weights_after_returns
from alpharank.portfolio.performance import performance_statistics

BENCHMARK_TICKER = "SPY.US"
WEIGHT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class ValuationWindow:
    """An exact close-to-close interval, with returns stored as fractions."""

    name: str
    start_date: date
    end_date: date


def _validate_holdings(holdings: pl.DataFrame) -> None:
    required = {"strategy", "ticker", "target_weight"}
    if not required.issubset(holdings.columns) or holdings.is_empty():
        raise ValueError(f"Frozen holdings require nonempty columns: {sorted(required)}")
    if holdings.select(pl.any_horizontal(pl.col(sorted(required)).is_null()).any()).item():
        raise ValueError("Frozen holding identifiers and weights cannot be null.")
    if holdings.select(pl.struct("strategy", "ticker").is_duplicated().any()).item():
        raise ValueError("Duplicate frozen holding key: strategy, ticker.")
    invalid_weight = ~pl.col("target_weight").is_finite() | (pl.col("target_weight") <= 0.0)
    if holdings.filter(invalid_weight).height:
        raise ValueError("Frozen holding weights must be finite and positive.")
    totals = holdings.group_by("strategy").agg(pl.col("target_weight").sum())
    if totals.filter((pl.col("target_weight") - 1.0).abs() > WEIGHT_TOLERANCE).height:
        raise ValueError("Frozen holding weights must sum to one per strategy.")


def _validated_prices(
    prices: pl.DataFrame, tickers: set[str], entry_date: date, end_date: date
) -> pl.DataFrame:
    required = {"ticker", "date", "adjusted_close"}
    if not required.issubset(prices.columns) or prices.schema["date"] != pl.Date:
        raise ValueError("Prices require ticker, date (Date), adjusted_close (positive USD).")
    selected = prices.filter(
        pl.col("ticker").is_in(sorted(tickers)) & pl.col("date").is_between(entry_date, end_date)
    ).select("ticker", "date", "adjusted_close")
    if selected.select(pl.struct("ticker", "date").is_duplicated().any()).item():
        raise ValueError("Duplicate price key: ticker, date.")
    invalid = (
        pl.col("adjusted_close").is_null()
        | ~pl.col("adjusted_close").is_finite()
        | (pl.col("adjusted_close") <= 0.0)
    )
    if selected.filter(invalid).height:
        raise ValueError("Every selected price must be finite, positive and nonnull.")
    benchmark_dates = set(selected.filter(pl.col("ticker") == BENCHMARK_TICKER)["date"].to_list())
    for ticker in sorted(tickers):
        dates = set(selected.filter(pl.col("ticker") == ticker)["date"].to_list())
        if not dates or dates != benchmark_dates:
            raise ValueError(f"Incomplete common price calendar for {ticker}.")
    if entry_date not in benchmark_dates or end_date not in benchmark_dates:
        raise ValueError("Entry and end date must exist in the common price calendar.")
    return selected


def _value_window(
    weights: dict[str, float],
    prices: dict[tuple[str, date], float],
    entry_date: date,
    window: ValuationWindow,
) -> float:
    entry_returns = {
        ticker: prices[ticker, window.start_date] / prices[ticker, entry_date] - 1.0
        for ticker in weights
    }
    drifted = drifted_weights_after_returns(weights, entry_returns)
    return fsum(
        drifted[ticker]
        * (prices[ticker, window.end_date] / prices[ticker, window.start_date] - 1.0)
        for ticker in weights
    )


def value_frozen_holdings(
    holdings: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    entry_date: date,
    windows: Sequence[ValuationWindow],
) -> pl.DataFrame:
    """Value buy-and-hold baskets against SPY on strict common close dates.

    Holdings have grain strategy/ticker with positive target_weight summing to
    one; prices have grain ticker/date with positive adjusted_close. All used
    tickers must share SPY's complete observed calendar. No price is filled and
    no holding is dropped. Subperiod weights drift from the injected entry date.
    Returns are gross of trading fees: this valuation creates no orders and
    does not invent entry costs or rebalance an expired signal. Raises
    ValueError on missing prices, invalid dates, weights or duplicate keys.
    """
    _validate_holdings(holdings)
    if not windows or len({window.name for window in windows}) != len(windows):
        raise ValueError("Valuation windows must be nonempty with unique names.")
    if any(not entry_date <= window.start_date < window.end_date for window in windows):
        raise ValueError("Each window must satisfy entry_date <= start_date < end_date.")
    tickers = set(holdings["ticker"].to_list()) | {BENCHMARK_TICKER}
    selected = _validated_prices(prices, tickers, entry_date, max(w.end_date for w in windows))
    available_dates = set(selected["date"].to_list())
    if any(
        w.start_date not in available_dates or w.end_date not in available_dates for w in windows
    ):
        raise ValueError("Each window endpoint must exist in the common price calendar.")
    lookup = {(ticker, day): float(price) for ticker, day, price in selected.iter_rows()}
    rows: list[dict[str, object]] = []
    for strategy_frame in holdings.sort("strategy", "ticker").partition_by("strategy"):
        strategy = str(strategy_frame["strategy"][0])
        weights = dict(strategy_frame.select("ticker", "target_weight").iter_rows())
        for window in windows:
            total_return = _value_window(weights, lookup, entry_date, window)
            benchmark_return = (
                lookup[BENCHMARK_TICKER, window.end_date]
                / lookup[BENCHMARK_TICKER, window.start_date]
                - 1.0
            )
            rows.append(
                {
                    "strategy": strategy,
                    "window": window.name,
                    "entry_date": entry_date,
                    "start_date": window.start_date,
                    "end_date": window.end_date,
                    "total_return": total_return,
                    "benchmark_return": benchmark_return,
                    "excess_return": total_return - benchmark_return,
                    "relative_wealth": (1.0 + total_return) / (1.0 + benchmark_return),
                }
            )
    return pl.DataFrame(rows)


def _valuation_curve(
    strategy: str,
    weights: dict[str, float],
    lookup: dict[tuple[str, date], float],
    entry_date: date,
    window: ValuationWindow,
) -> pl.DataFrame:
    dates = sorted(
        day
        for ticker, day in lookup
        if ticker == BENCHMARK_TICKER and window.start_date <= day <= window.end_date
    )
    returns = np.asarray(
        [
            _value_window(
                weights, lookup, entry_date, ValuationWindow(window.name, window.start_date, day)
            )
            for day in dates
        ],
        dtype=np.float64,
    )
    wealth = 1.0 + returns
    spy_wealth = np.asarray(
        [
            lookup[BENCHMARK_TICKER, day] / lookup[BENCHMARK_TICKER, window.start_date]
            for day in dates
        ],
        dtype=np.float64,
    )
    # Only the frequency-independent drawdown is used; monthly annualizations are not exposed.
    statistics = performance_statistics(np.r_[0.0, wealth[1:] / wealth[:-1] - 1.0])
    return pl.DataFrame(
        {
            "strategy": [strategy] * len(dates),
            "window": [window.name] * len(dates),
            "date": dates,
            "wealth": wealth,
            "total_return": returns,
            "drawdown": wealth / np.maximum.accumulate(wealth) - 1.0,
            "relative_wealth": wealth / spy_wealth,
            "max_drawdown": [statistics["max_drawdown"]] * len(dates),
        }
    )


def frozen_valuation_curves(
    holdings: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    entry_date: date,
    windows: Sequence[ValuationWindow],
) -> pl.DataFrame:
    """Project validated frozen valuations into daily wealth and drawdown paths.

    Shares the exact input/calendar contract and gross-return convention of
    value_frozen_holdings. Each window includes its starting close at wealth 1,
    preserves entry-date weight drift, and includes SPY. No annualized KPI is
    inferred from daily observations. Grain: strategy/window/date.
    """
    value_frozen_holdings(holdings, prices, entry_date=entry_date, windows=windows)
    benchmark_label = "SPY · Total return"
    if benchmark_label in holdings["strategy"].to_list():
        raise ValueError("Frozen strategy labels must not collide with the SPY benchmark.")
    selected = _validated_prices(
        prices,
        set(holdings["ticker"].to_list()) | {BENCHMARK_TICKER},
        entry_date,
        max(window.end_date for window in windows),
    )
    lookup = {(str(ticker), day): float(price) for ticker, day, price in selected.iter_rows()}
    allocations = {
        str(frame["strategy"][0]): dict(frame.select("ticker", "target_weight").iter_rows())
        for frame in holdings.sort("strategy", "ticker").partition_by("strategy")
    }
    allocations[benchmark_label] = {BENCHMARK_TICKER: 1.0}
    return pl.concat(
        [
            _valuation_curve(strategy, weights, lookup, entry_date, window)
            for window in windows
            for strategy, weights in allocations.items()
        ]
    ).sort("window", "strategy", "date")
