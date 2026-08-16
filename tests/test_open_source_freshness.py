from __future__ import annotations

import polars as pl
import pytest

from alpharank.data.open_source.freshness import (
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
