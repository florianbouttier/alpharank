"""Value sealed monthly baskets at observed daily closes without reranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from alpharank.portfolio.daily.contracts import DailyLedgerResult

PARITY_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class _MonthMarket:
    month: date
    anchor: date
    sessions: tuple[date, ...]
    prices: dict[tuple[str, date], float]


@dataclass(frozen=True, slots=True)
class _MonthlyTerms:
    strategy: str
    net_return: float
    cost: float
    turnover: float
    n_positions: int
    maximum_position_weight: float | None
    maximum_sector_weight: float | None


def _monthly_terms(row: dict[str, Any]) -> _MonthlyTerms:
    """Convert the schema-validated Polars row boundary to typed financial inputs."""
    return _MonthlyTerms(
        str(row["strategy"]),
        float(row["net_return"]),
        float(row["transaction_cost"]),
        float(row.get("turnover", 0.0)),
        int(row.get("n_positions", 1)),
        float(row["maximum_position_weight"])
        if row.get("maximum_position_weight") is not None
        else None,
        float(row["maximum_sector_weight"])
        if row.get("maximum_sector_weight") is not None
        else None,
    )


def _validate_inputs(holdings: pl.DataFrame, monthly: pl.DataFrame, prices: pl.DataFrame) -> None:
    required = {"strategy", "holding_month", "ticker", "target_weight"}
    if not required.issubset(holdings.columns):
        raise ValueError("Daily holdings require strategy, holding_month, ticker, target_weight.")
    if not {"strategy", "holding_month", "net_return", "transaction_cost"}.issubset(
        monthly.columns
    ):
        raise ValueError("Daily valuation requires sealed monthly returns and transaction costs.")
    if not {"ticker", "date", "adjusted_close"}.issubset(prices.columns):
        raise ValueError("Daily prices require ticker, date and adjusted_close.")
    for frame, keys in [
        (holdings, ["strategy", "holding_month", "ticker"]),
        (monthly, ["strategy", "holding_month"]),
        (prices, ["ticker", "date"]),
    ]:
        if frame.select(keys).is_duplicated().any():
            raise ValueError("Duplicate daily valuation input key: " + ", ".join(keys))
    if prices.schema["date"] != pl.Date or monthly.schema["holding_month"] != pl.Date:
        raise ValueError("Daily valuation requires native Date input columns.")
    if holdings.schema["holding_month"] != pl.Date:
        raise ValueError("Daily holdings require a native Date holding_month.")
    _validate_weights(holdings)
    _validate_monthly(monthly)


def _validate_weights(holdings: pl.DataFrame) -> None:
    weights = holdings["target_weight"].to_numpy()
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Daily holdings require finite non-negative weights.")
    sums = holdings.group_by("strategy", "holding_month").agg(pl.col("target_weight").sum())
    if not np.allclose(sums["target_weight"].to_numpy(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("Daily holdings weights must sum to one for every strategy/month.")


def _validate_monthly(monthly: pl.DataFrame) -> None:
    for field in ["net_return", "transaction_cost"]:
        if not np.isfinite(monthly[field].to_numpy()).all():
            raise ValueError("Non-finite sealed monthly field: " + field)
    if monthly.filter((pl.col("net_return") <= -1) | (pl.col("transaction_cost") < 0)).height:
        raise ValueError(
            "Daily valuation requires positive terminal wealth and non-negative costs."
        )


def _market_months(
    monthly: pl.DataFrame, prices: pl.DataFrame, benchmark_ticker: str
) -> list[_MonthMarket]:
    benchmark = prices.filter(pl.col("ticker") == benchmark_ticker).sort("date")
    if benchmark.is_empty():
        raise ValueError("Daily valuation requires the explicit benchmark calendar.")
    grouped = {
        key[0]: group
        for key, group in benchmark.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .partition_by("month", as_dict=True)
        .items()
    }
    lookup = {
        (str(ticker), day): float(value) if value is not None else float("nan")
        for ticker, day, value in prices.select("ticker", "date", "adjusted_close").iter_rows()
    }
    result = []
    for month in sorted(monthly["holding_month"].unique().to_list()):
        previous = (
            date(month.year - 1, 12, 1)
            if month.month == 1
            else date(month.year, month.month - 1, 1)
        )
        if month not in grouped or previous not in grouped:
            raise ValueError(f"Benchmark lacks the month or its preceding anchor: {month}.")
        anchor = max(grouped[previous]["date"].to_list())
        sessions = tuple(grouped[month]["date"].to_list())
        result.append(_MonthMarket(month, anchor, sessions, lookup))
    return result


def _month_values(
    basket: pl.DataFrame, market: _MonthMarket, cost: float
) -> tuple[list[float], list[dict[str, object]]]:
    names = basket["ticker"].to_list()
    weights = basket["target_weight"].to_numpy()
    anchors = np.asarray([market.prices.get((name, market.anchor), float("nan")) for name in names])
    if not np.isfinite(anchors).all() or np.any(anchors <= 0):
        raise ValueError(f"Missing or invalid starting price at {market.anchor}: {names}.")
    values, gaps = [1.0], []
    for day in market.sessions:
        quotes = np.asarray([market.prices.get((name, day), float("nan")) for name in names])
        for index, name in enumerate(names):
            if not math.isfinite(quotes[index]):
                gaps.append({"ticker": name, "valuation_date": day, "holding_month": market.month})
            elif quotes[index] <= 0:
                raise ValueError(f"Non-positive selected daily price: {name}, {day}.")
        # Fixed month-entry cash debit preserves gross_return - cost, not (1-cost)*(1+gross).
        value = float(np.dot(weights, quotes / anchors) - cost)
        if math.isfinite(value) and value <= 0:
            raise ValueError("Non-positive portfolio wealth in daily valuation.")
        values.append(value)
    return values, gaps


def _ledger_rows(
    strategy: str, market: _MonthMarket, values: list[float], sealed: _MonthlyTerms, capital: float
) -> list[dict[str, object]]:
    rows = []
    for index, day in enumerate((market.anchor, *market.sessions)):
        value = values[index]
        previous = values[index - 1] if index else 1.0
        rows.append(
            {
                "strategy": strategy,
                "valuation_date": day,
                "holding_month": market.month,
                "wealth": capital * value if math.isfinite(value) else None,
                "daily_return": value / previous - 1
                if math.isfinite(value) and math.isfinite(previous)
                else None,
                "month_factor": value if math.isfinite(value) else None,
                "sealed_monthly_return": sealed.net_return,
                "transaction_cost": sealed.cost if index == 1 else 0.0,
                "turnover": sealed.turnover if index == 1 else 0.0,
                "n_positions": sealed.n_positions,
                "maximum_position_weight": sealed.maximum_position_weight,
                "maximum_sector_weight": sealed.maximum_sector_weight,
            }
        )
    return rows


def build_daily_ledger(
    holdings: pl.DataFrame,
    monthly: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    benchmark_ticker: str = "SPY.US",
    benchmark_strategy: str | None = None,
) -> DailyLedgerResult:
    """Value each frozen basket daily, keep gaps and require observed month-end parity.

    Costs are debited on the first holding session as a fixed fraction of entry
    capital. Intramonth weights drift; no daily rebalance or price fill occurs.
    Monthly sealed capital anchors the next month even if an intermediate close
    is unavailable; missing daily closes and associated risk remain explicit.
    """
    _validate_inputs(holdings, monthly, prices)
    monthly = monthly.sort("holding_month", "strategy")
    references = set(monthly["strategy"]) - set(holdings["strategy"])
    reference = benchmark_strategy or (next(iter(references)) if len(references) == 1 else "")
    if not reference or reference not in references:
        raise ValueError("Daily valuation requires exactly identified benchmark strategy.")
    baskets = holdings.partition_by(["strategy", "holding_month"], as_dict=True)
    capitals = dict.fromkeys(monthly["strategy"].unique().to_list(), 1.0)
    first_month = min(monthly["holding_month"].to_list())
    rows, gaps, reconciliation = [], [], []
    for market in _market_months(monthly, prices, benchmark_ticker):
        for raw in monthly.filter(pl.col("holding_month") == market.month).iter_rows(named=True):
            sealed = _monthly_terms(raw)
            strategy = sealed.strategy
            basket = (
                pl.DataFrame({"ticker": [benchmark_ticker], "target_weight": [1.0]})
                if strategy == reference
                else baskets.get((strategy, market.month))
            )
            if basket is None:
                raise ValueError(f"Missing frozen basket: {strategy}, {market.month}.")
            values, missing = _month_values(basket, market, sealed.cost)
            expected = sealed.net_return
            error = abs(values[-1] - 1 - expected) if math.isfinite(values[-1]) else None
            if error is not None and error > PARITY_TOLERANCE:
                raise ValueError(
                    f"Daily/monthly return mismatch: {strategy}, {market.month}, error={error}."
                )
            part = _ledger_rows(strategy, market, values, sealed, capitals[strategy])
            rows.extend(part if market.month == first_month else part[1:])
            capitals[strategy] *= 1 + expected
            gaps.extend(missing)
            reconciliation.append(
                {
                    "strategy": strategy,
                    "holding_month": market.month,
                    "sealed_return": expected,
                    "observed_return": values[-1] - 1 if error is not None else None,
                    "absolute_error": error,
                    "status": "matched" if error is not None else "missing_daily_endpoint",
                }
            )
    ledger = pl.from_dicts(rows, infer_schema_length=None).sort("strategy", "valuation_date")
    gap_frame = (
        pl.from_dicts(
            gaps, schema={"ticker": pl.String, "valuation_date": pl.Date, "holding_month": pl.Date}
        )
        .unique()
        .sort("valuation_date", "ticker")
    )
    return DailyLedgerResult(
        ledger,
        pl.from_dicts(reconciliation, infer_schema_length=None),
        gap_frame,
        {
            "missing_ticker_dates": gap_frame.height,
            "missing_valuations": ledger["wealth"].null_count(),
            "ledger_rows": ledger.height,
        },
    )
