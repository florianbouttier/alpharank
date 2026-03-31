from __future__ import annotations

import polars as pl

from alpharank.data.open_source.consolidation import FinancialSourceInput, consolidate_financial_sources


def test_consolidate_financial_sources_keeps_default_priority_for_non_share_metrics() -> None:
    sec = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2025-12-31"],
            "filing_date": ["2026-01-30"],
            "value": [100.0],
            "source": ["sec_companyfacts"],
            "source_label": ["Revenues"],
        }
    )
    yahoo = sec.with_columns(
        [
            pl.lit(90.0).alias("value"),
            pl.lit("yfinance").alias("source"),
            pl.lit("Total Revenue").alias("source_label"),
        ]
    )

    consolidated, _, _ = consolidate_financial_sources(
        [
            FinancialSourceInput("sec_companyfacts", sec, 1),
            FinancialSourceInput("yfinance", yahoo, 4),
        ]
    )

    assert consolidated["selected_source"].to_list() == ["sec_companyfacts"]


def test_consolidate_financial_sources_overrides_share_outlier_with_yahoo() -> None:
    sec = pl.DataFrame(
        {
            "ticker": ["ACN.US"],
            "statement": ["shares"],
            "metric": ["outstanding_shares"],
            "date": ["2025-08-31"],
            "filing_date": ["2025-10-10"],
            "value": [302_358.0],
            "source": ["sec_filing"],
            "source_label": ["SummedStatementClassOfStockAxisMembers"],
        }
    )
    yahoo = sec.with_columns(
        [
            pl.lit(621_855_922.0).alias("value"),
            pl.lit("yfinance").alias("source"),
            pl.lit("Ordinary Shares Number").alias("source_label"),
            pl.lit(None).cast(pl.Utf8).alias("filing_date"),
        ]
    )

    consolidated, lineage, _ = consolidate_financial_sources(
        [
            FinancialSourceInput("sec_filing", sec, 2),
            FinancialSourceInput("yfinance", yahoo, 4),
        ]
    )

    assert consolidated["selected_source"].to_list() == ["yfinance"]
    assert consolidated["value"].to_list() == [621_855_922.0]
    assert lineage["source"].to_list() == ["sec_filing", "yfinance"]
