from __future__ import annotations

import polars as pl
import pytest

from alpharank.data.quality.freshness import (
    build_data_freshness_summary,
    validate_data_freshness,
)


def _summary() -> dict[str, object]:
    return build_data_freshness_summary(
        prices=pl.DataFrame(
            {
                "ticker": ["AAPL.US"],
                "date": ["2026-08-12"],
                "adjusted_close": [220.0],
            }
        ),
        benchmark_prices=pl.DataFrame({"date": ["2026-08-12"]}),
        financials=pl.DataFrame(
            {
                "date": ["2026-06-30"],
                "filing_date": ["2026-08-07"],
            }
        ),
        earnings_sec_calendar=pl.DataFrame(
            {
                "period_end": ["2026-06-30"],
                "reportDate": ["2026-08-07"],
            }
        ),
        constituents=pl.DataFrame({"Date": ["2026-08-01"], "Ticker": ["AAPL"]}),
    )


def test_freshness_summary_separates_period_end_from_filing_date() -> None:
    summary = _summary()

    assert summary["financials"]["max_fiscal_period_end"] == "2026-06-30"
    assert summary["financials"]["max_sec_filing_date"] == "2026-08-07"
    validate_data_freshness(summary, expected_through="2026-08-13")


def test_freshness_gate_rejects_stale_sec_and_membership_layers() -> None:
    summary = _summary()
    summary["financials"]["max_sec_filing_date"] = "2026-06-05"
    summary["sp500_constituents"]["max_membership_month"] = "2026-07-01"

    with pytest.raises(RuntimeError, match="financials.max_sec_filing_date") as error:
        validate_data_freshness(summary, expected_through="2026-08-13")

    assert "sp500_constituents.max_membership_month" in str(error.value)


def test_confirmed_terminal_member_does_not_stale_the_refreshable_universe() -> None:
    inputs = {
        "prices": pl.DataFrame(
            {
                "ticker": ["AAPL.US", "EA.US"],
                "date": ["2026-08-19", "2026-08-10"],
                "adjusted_close": [230.0, 210.0],
            }
        ),
        "benchmark_prices": pl.DataFrame({"date": ["2026-08-19"]}),
        "financials": pl.DataFrame(
            {"date": ["2026-06-30"], "filing_date": ["2026-08-19"]}
        ),
        "earnings_sec_calendar": pl.DataFrame(
            {"period_end": ["2026-06-30"], "reportDate": ["2026-08-19"]}
        ),
        "constituents": pl.DataFrame(
            {
                "Date": ["2026-08-01", "2026-08-01"],
                "Ticker": ["AAPL", "EA"],
            }
        ),
    }

    stale = build_data_freshness_summary(**inputs)
    with pytest.raises(
        RuntimeError,
        match="current_member_latest_common_market_date=2026-08-10",
    ):
        validate_data_freshness(stale, expected_through="2026-08-20")

    reviewed = build_data_freshness_summary(
        **inputs,
        terminal_tickers=("EA.US",),
    )

    assert reviewed["prices"]["current_member_count"] == 2
    assert reviewed["prices"]["current_member_refreshable_count"] == 1
    assert reviewed["prices"]["current_member_terminal_exclusion_count"] == 1
    assert reviewed["prices"]["current_member_terminal_exclusion_examples"] == [
        "EA.US"
    ]
    assert (
        reviewed["prices"]["current_member_latest_common_market_date"]
        == "2026-08-19"
    )
    validate_data_freshness(reviewed, expected_through="2026-08-20")
