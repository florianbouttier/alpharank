from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

from alpharank.data.open_source.sec_only import (
    build_sec_only_earnings,
    build_sec_only_financials,
    build_sec_only_general_reference_from_raw_lineage,
)


def test_build_sec_only_general_reference_from_raw_lineage_ignores_yahoo_fields() -> None:
    raw_lineage = pl.DataFrame(
        {
            "ticker": ["NEM.US"],
            "name": ["Newmont Corp"],
            "exchange": ["NYSE"],
            "cik": ["0001164727"],
            "source": ["open_source_general"],
            "Sector": ["Basic Materials"],
            "industry": ["Gold"],
            "sector_source": ["yfinance"],
            "sector_raw_value": ["Materials"],
            "sic": ["1040"],
            "sic_description": ["Gold Ores"],
            "mapping_rule": ["yfinance:sector"],
            "selected_name_source": ["yfinance"],
            "selected_exchange_source": ["yfinance"],
            "yahoo_name": ["Newmont Corp"],
            "yahoo_exchange": ["NYSE"],
            "yahoo_sector": ["Materials"],
            "yahoo_industry": ["Gold"],
            "sec_name": ["NEWMONT CORP"],
            "sec_exchange": ["NYSE"],
            "sec_cik": ["1164727"],
            "sec_sic": ["1040"],
            "sec_sic_description": ["Gold Ores"],
        }
    )

    general_reference, lineage = build_sec_only_general_reference_from_raw_lineage(raw_lineage)

    assert general_reference["name"].to_list() == ["NEWMONT CORP"]
    assert general_reference["Sector"].to_list() == ["Basic Materials"]
    assert general_reference["sector_source"].to_list() == ["sec_sic"]
    assert lineage["selected_name_source"].to_list() == ["sec_mapping"]
    assert lineage["yahoo_name"].to_list() == [None]


def test_build_sec_only_earnings_keeps_sec_actuals_and_nulls_market_fields() -> None:
    sec_calendar = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "period_end": ["2025-03-29"],
            "reportDate": ["2025-05-01"],
            "earningsDatetime": ["2025-05-01 20:00:00"],
            "accession_number": ["0000320193-25-000001"],
            "form": ["10-Q"],
            "fiscal_period": ["Q2"],
            "fiscal_year": [2025],
            "source": ["sec_submissions"],
            "source_label": ["reportDate"],
        }
    )
    sec_actuals = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "period_end": ["2025-03-29"],
            "reportDate": ["2025-05-01"],
            "epsActual": [1.53],
            "source": ["sec_companyfacts"],
            "source_label": ["EarningsPerShareDiluted"],
            "form": ["10-Q"],
            "fiscal_period": ["Q2"],
            "fiscal_year": [2025],
        }
    )

    consolidated, lineage, long_frame = build_sec_only_earnings(
        sec_calendar=sec_calendar,
        sec_actuals=sec_actuals,
    )

    assert consolidated["epsActual"].to_list() == [1.53]
    assert consolidated["epsEstimate"].to_list() == [None]
    assert consolidated["surprisePercent"].to_list() == [None]
    assert consolidated["actual_source"].to_list() == ["sec_companyfacts"]
    assert consolidated["selected_source"].to_list() == ["sec_submissions+sec_companyfacts"]
    assert lineage["sec_epsActual"].to_list() == [1.53]
    assert lineage["yahoo_epsActual"].to_list() == [None]
    assert long_frame["source"].to_list() == ["open_source_earnings"]


def test_build_sec_only_financials_prefers_filing_for_shares_and_drops_absurd_companyfacts_values() -> None:
    sec_companyfacts = pl.DataFrame(
        {
            "ticker": ["UPS.US", "UPS.US", "TFC.US", "TFC.US"],
            "statement": ["shares", "shares", "shares", "shares"],
            "metric": ["outstanding_shares"] * 4,
            "date": ["2025-04-15", "2025-06-30", "2008-12-31", "2009-03-31"],
            "filing_date": ["2025-05-01", "2025-08-01", "2009-08-13", "2009-11-09"],
            "value": [113_070_725.0, 847_569_642.0, 559_248_000_000.0, 687_617_017.0],
            "source": ["sec_companyfacts"] * 4,
            "source_label": ["EntityCommonStockSharesOutstanding"] * 4,
            "accession_number": [None] * 4,
            "form": ["10-Q"] * 4,
            "fiscal_period": ["Q1", "Q2", "Q2", "Q3"],
            "fiscal_year": [2025, 2025, 2009, 2009],
        }
    )
    sec_filing = pl.DataFrame(
        {
            "ticker": ["UPS.US", "UPS.US"],
            "statement": ["shares", "shares"],
            "metric": ["outstanding_shares", "outstanding_shares"],
            "date": ["2025-03-31", "2025-06-30"],
            "filing_date": ["2025-05-01", "2025-08-01"],
            "value": [846_797_806.0, 847_569_642.0],
            "source": ["sec_filing", "sec_filing"],
            "source_label": ["SummedStatementClassOfStockAxisMembers"] * 2,
            "accession_number": ["0001", "0002"],
            "form": ["10-Q", "10-Q"],
            "fiscal_period": ["Q1", "Q2"],
            "fiscal_year": [2025, 2025],
        }
    )

    consolidated, lineage, _ = build_sec_only_financials(
        sec_companyfacts=sec_companyfacts,
        sec_filing=sec_filing,
    )

    ups_q1 = consolidated.filter(
        (pl.col("ticker") == "UPS.US")
        & (pl.col("metric") == "outstanding_shares")
        & (pl.col("selected_fiscal_year") == 2025)
        & (pl.col("selected_fiscal_period") == "Q1")
    )
    assert ups_q1.height == 1
    assert ups_q1["date"].to_list() == ["2025-03-31"]
    assert ups_q1["selected_source"].to_list() == ["sec_filing"]

    tfc_rows = consolidated.filter(pl.col("ticker") == "TFC.US")
    assert tfc_rows["date"].to_list() == ["2009-03-31"]
    assert tfc_rows["value"].to_list() == [687_617_017.0]
    assert (
        lineage.filter(
            (pl.col("ticker") == "UPS.US")
            & (pl.col("selected_fiscal_year") == 2025)
            & (pl.col("selected_fiscal_period") == "Q1")
            & (pl.col("date") == "2025-03-31")
        ).height
        == 1
    )


def test_build_quarterly_presence_uses_separate_financial_and_eps_grids() -> None:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "open_source" / "build_sec_quality_dashboard.py"
    spec = importlib.util.spec_from_file_location("build_sec_quality_dashboard", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    financials = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US", "AAA.US", "AAA.US", "AAA.US", "AAA.US"],
            "metric": ["revenue", "net_income", "outstanding_shares", "revenue", "net_income", "outstanding_shares"],
            "selected_fiscal_year": [2025, 2025, 2025, 2025, 2025, 2025],
            "selected_fiscal_period": ["Q1", "Q1", "Q1", "Q3", "Q3", "Q3"],
        }
    )
    earnings = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "fiscal_year": [2025, 2025],
            "fiscal_period": ["Q1", "Q2"],
            "epsActual": [1.0, 1.1],
        }
    )
    general = pl.DataFrame(
        {
            "Code": ["AAA"],
            "Sector": ["Technology"],
            "Industry": ["Software"],
        }
    )

    quarterly_presence = module._build_quarterly_presence(
        financials=financials,
        earnings=earnings,
        general=general,
    )
    ticker_metric_holes = module._build_ticker_metric_holes(quarterly_presence=quarterly_presence)

    revenue_holes = ticker_metric_holes.filter(
        (pl.col("ticker") == "AAA.US") & (pl.col("metric") == "revenue")
    ).row(0, named=True)
    eps_holes = ticker_metric_holes.filter(
        (pl.col("ticker") == "AAA.US") & (pl.col("metric") == "epsActual")
    ).row(0, named=True)

    assert revenue_holes["expected_quarters"] == 3
    assert revenue_holes["hole_count"] == 1
    assert revenue_holes["sample_missing_dates"] == "2025 Q2"
    assert eps_holes["expected_quarters"] == 2
    assert eps_holes["hole_count"] == 0
