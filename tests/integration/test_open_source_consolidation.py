from __future__ import annotations

import polars as pl

from alpharank.data.open_source.consolidation import (
    FinancialSourceInput,
    consolidate_financial_sources,
    consolidate_financial_sources_with_share_quality,
)


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


def test_consolidate_financial_sources_preserves_accession_number_in_selected_lineage() -> None:
    filing = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["income_statement"],
            "metric": ["revenue"],
            "date": ["2025-03-31"],
            "filing_date": ["2025-05-01"],
            "value": [95.0],
            "source": ["sec_filing"],
            "source_label": ["Revenues"],
            "accession_number": ["0000320193-25-000073"],
        }
    )

    consolidated, lineage, _ = consolidate_financial_sources(
        [
            FinancialSourceInput("sec_filing", filing, 2),
        ]
    )

    assert consolidated["selected_accession_number"].to_list() == ["0000320193-25-000073"]
    assert lineage["selected_accession_number"].to_list() == ["0000320193-25-000073"]
def test_share_quality_reselects_fallback_after_cross_source_scale_mismatch() -> None:
    primary = pl.DataFrame(
        {
            "ticker": ["BRK-B.US"] * 3,
            "statement": ["shares"] * 3,
            "metric": ["outstanding_shares"] * 3,
            "date": ["2011-01-31", "2011-02-28", "2011-03-31"],
            "filing_date": ["2011-02-15", "2011-03-15", "2011-04-15"],
            "value": [1_070_000_000.0, 943_242.0, 1_068_000_000.0],
            "source": ["sec_companyfacts"] * 3,
            "source_label": ["shares"] * 3,
        }
    )
    fallback = primary.with_columns(
        pl.Series("value", [1_071_000_000.0, 1_069_000_000.0, 1_067_000_000.0]),
        pl.lit("sec_filing").alias("source"),
    )

    consolidated, _, _, report = consolidate_financial_sources_with_share_quality(
        [
            FinancialSourceInput("sec_companyfacts", primary, 1),
            FinancialSourceInput("sec_filing", fallback, 2),
        ]
    )

    selected = consolidated.filter(pl.col("date") == "2011-02-28")
    assert selected["value"].item() == 1_069_000_000.0
    assert selected["selected_source"].item() == "sec_filing"
    assert report["selection_passes"] == 2
    assert report["unresolved_selected_rows"] == 0
