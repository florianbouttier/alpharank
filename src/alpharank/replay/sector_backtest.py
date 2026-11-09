"""Full-calendar replay for causal one-name-per-sector allocations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl

from alpharank.portfolio.performance import annual_returns, portfolio_period_statistics
from alpharank.portfolio.simulation import simulate_weighted_portfolio

SECTOR_NATIVE = "Boosting Top 5 | 1 per sector"
SECTOR_TREND = "Boosting Top 5 | Causal trend | 1 per sector"
REFERENCE_STRATEGIES = (
    "Legacy",
    "Boosting Top 5",
    "Boosting Top 5 | Causal trend",
    "SPY total return",
)
DEFAULT_START_YEARS = (2011, 2015, 2020, 2023, 2024, 2025, 2026)


@dataclass(frozen=True, slots=True)
class SectorBacktestResult:
    """Structured outputs calculated by the canonical portfolio engines."""

    holdings: pl.DataFrame
    monthly_returns: pl.DataFrame
    performance_summary: pl.DataFrame
    annual_returns: pl.DataFrame
    start_year_metrics: pl.DataFrame


def build_sector_backtest(
    native_holdings: pl.DataFrame,
    trend_holdings: pl.DataFrame,
    reference_monthly: pl.DataFrame,
    *,
    transaction_cost_bps: float = 10.0,
    start_years: Sequence[int] = DEFAULT_START_YEARS,
) -> SectorBacktestResult:
    """Replay two causal sector allocations on the reference common calendar.

    The score-only tail is accepted only when it is trailing, fully null and
    explicitly marked ``horizon_pending``. Any partial or mature missing return
    fails before simulation. Month-level timing is retained because this replay
    consumes the already-audited immutable Boosting predictions rather than
    reconstructing their feature timestamps.
    """

    native = _prepare_sector_holdings(native_holdings, expected_strategy=SECTOR_NATIVE)
    trend = _prepare_sector_holdings(trend_holdings, expected_strategy=SECTOR_TREND)
    _validate_matching_calendars(native, trend)
    sector_holdings = pl.concat([native, trend], how="diagonal_relaxed")
    sector_monthly = simulate_weighted_portfolio(
        sector_holdings,
        transaction_cost_bps=transaction_cost_bps,
        missing_return_policy="raise",
        causal_timing_policy="legacy_month_only",
    )
    references = _select_reference_monthly(reference_monthly)
    _validate_reference_calendar(sector_monthly, references)
    monthly = pl.concat([references, sector_monthly], how="diagonal_relaxed").sort(
        ["strategy", "holding_month"]
    )
    annual = _annual_return_table(monthly)
    summary = _performance_summary(monthly, annual)
    start_metrics = _start_year_metrics(monthly, start_years=start_years)
    return SectorBacktestResult(
        holdings=sector_holdings,
        monthly_returns=monthly,
        performance_summary=summary,
        annual_returns=annual,
        start_year_metrics=start_metrics,
    )


def _prepare_sector_holdings(frame: pl.DataFrame, *, expected_strategy: str) -> pl.DataFrame:
    required = {
        "strategy",
        "decision_month",
        "holding_month",
        "ticker",
        "target_weight",
        "realized_return",
        "benchmark_return",
        "target_status_1m",
        "Sector",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Sector holdings are missing columns: " + ", ".join(missing))
    strategies = frame["strategy"].unique().to_list()
    if strategies != [expected_strategy]:
        raise ValueError(f"Expected strategy={expected_strategy!r}, found={strategies!r}.")
    audit = (
        frame.group_by("decision_month")
        .agg(
            pl.len().alias("rows"),
            pl.col("realized_return").is_finite().sum().alias("finite_returns"),
            pl.col("benchmark_return").is_finite().sum().alias("finite_benchmark"),
            pl.col("target_status_1m").unique().alias("statuses"),
            pl.col("Sector").n_unique().alias("sector_count"),
        )
        .sort("decision_month")
    )
    invalid_selection = audit.filter((pl.col("rows") != 5) | (pl.col("sector_count") != 5))
    if not invalid_selection.is_empty():
        raise ValueError(f"Sector allocation is invalid: {invalid_selection.row(0, named=True)}")
    partial = audit.filter(
        ~(
            ((pl.col("finite_returns") == 5) & (pl.col("finite_benchmark") == 5))
            | ((pl.col("finite_returns") == 0) & (pl.col("finite_benchmark") == 0))
        )
    )
    if not partial.is_empty():
        raise ValueError(f"Partially missing portfolio month: {partial.row(0, named=True)}")
    pending = audit.filter(pl.col("finite_returns") == 0)
    completed = audit.filter(pl.col("finite_returns") == 5)
    if completed.is_empty():
        raise ValueError("Sector allocation has no completed return month.")
    if not pending.is_empty():
        invalid_pending = pending.filter(
            (pl.col("decision_month") <= completed["decision_month"].max())
            | (pl.col("statuses") != pl.lit(["horizon_pending"]))
        )
        if not invalid_pending.is_empty():
            raise ValueError(f"Non-trailing pending month: {invalid_pending.row(0, named=True)}")
    completed_months = completed.select("decision_month")
    return (
        frame.join(completed_months, on="decision_month", how="inner")
        .rename({"Sector": "sector"})
        .sort(["decision_month", "ticker"])
    )


def _validate_matching_calendars(left: pl.DataFrame, right: pl.DataFrame) -> None:
    left_months = left.select("decision_month").unique().sort("decision_month")
    right_months = right.select("decision_month").unique().sort("decision_month")
    if not left_months.equals(right_months):
        raise ValueError("Native and trend sector allocations do not share one calendar.")


def _select_reference_monthly(frame: pl.DataFrame) -> pl.DataFrame:
    selected = frame.filter(pl.col("strategy").is_in(REFERENCE_STRATEGIES))
    found = set(selected["strategy"].unique().to_list())
    missing = sorted(set(REFERENCE_STRATEGIES) - found)
    if missing:
        raise ValueError("Reference monthly returns are missing: " + ", ".join(missing))
    return selected.sort(["strategy", "holding_month"])


def _validate_reference_calendar(sector: pl.DataFrame, reference: pl.DataFrame) -> None:
    expected = sector.select("holding_month").unique().sort("holding_month")
    for strategy in REFERENCE_STRATEGIES:
        observed = (
            reference.filter(pl.col("strategy") == strategy)
            .select("holding_month")
            .unique()
            .sort("holding_month")
        )
        if not expected.equals(observed):
            raise ValueError(f"Calendar mismatch for reference strategy={strategy!r}.")


def _annual_return_table(monthly: pl.DataFrame) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for strategy in monthly["strategy"].unique(maintain_order=True).to_list():
        frame = monthly.filter(pl.col("strategy") == strategy).sort("holding_month")
        yearly = annual_returns(
            frame["net_return"].to_numpy(),
            holding_months=frame["holding_month"].to_list(),
        ).with_columns(pl.lit(strategy).alias("strategy"))
        parts.append(yearly.select("strategy", pl.all().exclude("strategy")))
    annual = pl.concat(parts).sort(["year", "strategy"])
    spy = annual.filter(pl.col("strategy") == "SPY total return").select(
        "year",
        pl.col("annual_return").alias("spy_annual_return"),
    )
    return annual.join(spy, on="year", how="left").with_columns(
        (pl.col("annual_return") - pl.col("spy_annual_return")).alias("annual_active_return"),
        ((1.0 + pl.col("annual_return")) / (1.0 + pl.col("spy_annual_return")) - 1.0).alias(
            "annual_relative_return"
        ),
    )


def _performance_summary(monthly: pl.DataFrame, annual: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for strategy in monthly["strategy"].unique(maintain_order=True).to_list():
        frame = monthly.filter(pl.col("strategy") == strategy).sort("holding_month")
        metrics = portfolio_period_statistics(
            frame["net_return"].to_numpy(),
            benchmark_returns=frame["benchmark_return"].to_numpy(),
            turnovers=frame["turnover"].to_numpy(),
            transaction_costs=frame["transaction_cost"].to_numpy(),
            position_counts=frame["n_positions"].to_numpy(),
            maximum_position_weights=frame["maximum_position_weight"].to_numpy(),
            maximum_sector_weights=frame["maximum_sector_weight"].to_numpy(),
        )
        yearly = annual.filter((pl.col("strategy") == strategy) & pl.col("is_full_calendar_year"))
        worst = _worst_year(yearly, return_column="annual_return")
        worst_relative = _worst_year(yearly, return_column="annual_relative_return")
        rows.append(
            {
                "strategy": strategy,
                "months": frame.height,
                **metrics,
                "full_calendar_years": yearly.height,
                "years_beating_spy": yearly.filter(pl.col("annual_active_return") > 0.0).height,
                "worst_full_year": worst[0],
                "worst_full_year_return": worst[1],
                "worst_relative_year": worst_relative[0],
                "worst_relative_year_return": worst_relative[1],
            }
        )
    return pl.DataFrame(rows).sort("strategy")


def _worst_year(yearly: pl.DataFrame, *, return_column: str) -> tuple[int, float]:
    if yearly.is_empty():
        return -1, float("nan")
    row = yearly.sort(return_column).row(0, named=True)
    return int(row["year"]), float(row[return_column])


def _start_year_metrics(
    monthly: pl.DataFrame,
    *,
    start_years: Sequence[int],
) -> pl.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for start_year in start_years:
        period = monthly.filter(pl.col("holding_month").dt.year() >= start_year)
        for strategy in period["strategy"].unique(maintain_order=True).to_list():
            frame = period.filter(pl.col("strategy") == strategy).sort("holding_month")
            metrics = portfolio_period_statistics(
                frame["net_return"].to_numpy(),
                benchmark_returns=frame["benchmark_return"].to_numpy(),
                turnovers=frame["turnover"].to_numpy(),
                transaction_costs=frame["transaction_cost"].to_numpy(),
                position_counts=frame["n_positions"].to_numpy(),
                maximum_position_weights=frame["maximum_position_weight"].to_numpy(),
                maximum_sector_weights=frame["maximum_sector_weight"].to_numpy(),
            )
            rows.append(
                {"start_year": start_year, "strategy": strategy, "months": frame.height, **metrics}
            )
    return pl.DataFrame(rows).sort(["start_year", "strategy"])
