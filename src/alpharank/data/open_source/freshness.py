from __future__ import annotations

from datetime import date
from typing import Any, Sequence

import polars as pl


def build_data_freshness_summary(
    *,
    prices: pl.DataFrame,
    benchmark_prices: pl.DataFrame,
    financials: pl.DataFrame,
    earnings_sec_calendar: pl.DataFrame,
    constituents: pl.DataFrame,
    terminal_tickers: Sequence[str] = (),
) -> dict[str, Any]:
    current_member_coverage = _current_member_price_coverage(
        prices,
        constituents,
        terminal_tickers=terminal_tickers,
    )
    return {
        "prices": {
            "max_market_date": _max_date(prices, "date"),
            **current_member_coverage,
        },
        "benchmark": {"max_market_date": _max_date(benchmark_prices, "date")},
        "financials": {
            "max_fiscal_period_end": _max_date(financials, "date"),
            "max_sec_filing_date": _max_date(financials, "filing_date"),
        },
        "earnings_sec_calendar": {
            "max_period_end": _max_date(earnings_sec_calendar, "period_end"),
            "max_sec_filing_date": _max_date(earnings_sec_calendar, "reportDate"),
        },
        "sp500_constituents": {
            "max_membership_month": _max_date(constituents, "Date"),
        },
    }


def validate_data_freshness(
    summary: dict[str, Any],
    *,
    expected_through: str | date,
    max_price_age_days: int = 7,
    max_sec_filing_age_days: int = 45,
) -> None:
    expected = date.fromisoformat(str(expected_through))
    failures: list[str] = []

    for layer in ("prices", "benchmark"):
        observed = _parse_required_date(summary[layer]["max_market_date"], f"{layer}.max_market_date")
        age = (expected - observed).days
        if age < 0 or age > max_price_age_days:
            failures.append(f"{layer}.max_market_date={observed} age_days={age} limit={max_price_age_days}")

    missing_members = int(summary["prices"].get("current_member_missing_count", 0))
    if missing_members:
        failures.append(
            "prices.current_member_missing_count="
            f"{missing_members} examples={summary['prices'].get('current_member_missing_examples', [])}"
        )
    latest_common = _parse_required_date(
        summary["prices"].get("current_member_latest_common_market_date"),
        "prices.current_member_latest_common_market_date",
    )
    common_age = (expected - latest_common).days
    if common_age < 0 or common_age > max_price_age_days:
        failures.append(
            "prices.current_member_latest_common_market_date="
            f"{latest_common} age_days={common_age} limit={max_price_age_days}"
        )

    filing_date = _parse_required_date(
        summary["financials"]["max_sec_filing_date"],
        "financials.max_sec_filing_date",
    )
    filing_age = (expected - filing_date).days
    if filing_age < 0 or filing_age > max_sec_filing_age_days:
        failures.append(
            "financials.max_sec_filing_date="
            f"{filing_date} age_days={filing_age} limit={max_sec_filing_age_days}"
        )

    membership_month = _parse_required_date(
        summary["sp500_constituents"]["max_membership_month"],
        "sp500_constituents.max_membership_month",
    )
    expected_month = expected.replace(day=1)
    if membership_month != expected_month:
        failures.append(
            f"sp500_constituents.max_membership_month={membership_month} expected={expected_month}"
        )

    if failures:
        raise RuntimeError("Data freshness gate failed before publication: " + "; ".join(failures))


def _max_date(frame: pl.DataFrame, column: str) -> str | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    value = frame.select(pl.col(column).cast(pl.Date, strict=False).max()).item()
    return str(value) if value is not None else None


def _parse_required_date(value: object, field: str) -> date:
    if value is None:
        raise RuntimeError(f"Data freshness gate failed before publication: missing {field}")
    return date.fromisoformat(str(value))


def _current_member_price_coverage(
    prices: pl.DataFrame,
    constituents: pl.DataFrame,
    *,
    terminal_tickers: Sequence[str] = (),
) -> dict[str, Any]:
    if prices.is_empty() or constituents.is_empty() or not {"Date", "Ticker"}.issubset(constituents.columns):
        return {
            "current_member_count": 0,
            "current_member_refreshable_count": 0,
            "current_member_terminal_exclusion_count": 0,
            "current_member_terminal_exclusion_examples": [],
            "current_member_missing_count": 0,
            "current_member_missing_examples": [],
            "current_member_latest_common_market_date": None,
        }
    normalized_constituents = constituents.select(
        pl.col("Date").cast(pl.Date, strict=False),
        (pl.col("Ticker").cast(pl.String).str.to_uppercase() + pl.lit(".US")).alias("ticker"),
    )
    latest_month = normalized_constituents.select(pl.col("Date").max()).item()
    all_members = (
        normalized_constituents.filter(pl.col("Date") == latest_month)
        .select("ticker")
        .unique()
    )
    normalized_terminal_tickers = sorted(
        {
            f"{str(ticker).upper().removesuffix('.US')}.US"
            for ticker in terminal_tickers
        }
    )
    terminal_members = all_members.filter(
        pl.col("ticker").is_in(normalized_terminal_tickers)
    )
    members = all_members.filter(
        ~pl.col("ticker").is_in(normalized_terminal_tickers)
    )
    coverage = (
        prices.select(
            pl.col("ticker").cast(pl.String).str.to_uppercase(),
            pl.col("date").cast(pl.Date, strict=False),
            pl.col("adjusted_close").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("adjusted_close").is_not_null())
        .group_by("ticker")
        .agg(pl.col("date").max().alias("max_date"))
    )
    joined = members.join(coverage, on="ticker", how="left")
    missing = joined.filter(pl.col("max_date").is_null())
    latest_common = joined.select(pl.col("max_date").min()).item()
    return {
        "current_member_count": all_members.height,
        "current_member_refreshable_count": members.height,
        "current_member_terminal_exclusion_count": terminal_members.height,
        "current_member_terminal_exclusion_examples": (
            terminal_members.get_column("ticker").sort().head(20).to_list()
        ),
        "current_member_missing_count": missing.height,
        "current_member_missing_examples": missing.get_column("ticker").head(20).to_list(),
        "current_member_latest_common_market_date": (
            str(latest_common) if latest_common is not None else None
        ),
    }
