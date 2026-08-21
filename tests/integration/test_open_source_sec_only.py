from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

from alpharank.data.sources.sec_only import (
    build_sec_only_earnings,
    build_sec_only_financials,
    build_sec_only_general_reference_from_raw_lineage,
    select_initial_sec_filing_versions,
)


def test_initial_sec_filing_version_is_selected_for_point_in_time_values() -> None:
    versions = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "statement": ["shares", "shares"],
            "metric": ["outstanding_shares", "outstanding_shares"],
            "date": ["2020-12-31", "2020-12-31"],
            "filing_date": ["2022-02-01", "2021-02-01"],
            "value": [110.0, 100.0],
            "source": ["sec_companyfacts", "sec_companyfacts"],
            "ingested_at": ["2026-08-16", "2026-08-16"],
        }
    )

    selected = select_initial_sec_filing_versions(versions)

    assert selected.select("filing_date", "value").rows() == [
        ("2021-02-01", 100.0)
    ]


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


def test_build_sec_only_earnings_derives_eps_from_sec_financials_when_missing() -> None:
    sec_calendar = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "period_end": ["2025-03-31"],
            "reportDate": ["2025-05-01"],
            "earningsDatetime": [None],
            "accession_number": ["0000001"],
            "form": ["10-Q"],
            "fiscal_period": ["Q1"],
            "fiscal_year": [2025],
            "source": ["sec_submissions"],
            "source_label": ["reportDate"],
        }
    )
    sec_actuals = pl.DataFrame(
        {
            "ticker": [],
            "period_end": [],
            "reportDate": [],
            "epsActual": [],
            "source": [],
            "source_label": [],
            "form": [],
            "fiscal_period": [],
            "fiscal_year": [],
        },
        schema={
            "ticker": pl.String,
            "period_end": pl.String,
            "reportDate": pl.String,
            "epsActual": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
        },
    )
    sec_financials = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "statement": ["income_statement", "shares"],
            "metric": ["net_income", "outstanding_shares"],
            "date": ["2025-03-31", "2025-03-31"],
            "filing_date": ["2025-05-01", "2025-05-01"],
            "value": [200.0, 100.0],
            "source": ["sec_companyfacts", "sec_filing"],
            "source_label": ["NetIncomeLoss", "EntityCommonStockSharesOutstanding"],
            "selected_source": ["sec_companyfacts", "sec_filing"],
            "selected_source_label": ["NetIncomeLoss", "EntityCommonStockSharesOutstanding"],
            "selected_accession_number": [None, "0001"],
            "selected_form": ["10-Q", "10-Q"],
            "selected_fiscal_period": ["Q1", "Q1"],
            "selected_fiscal_year": [2025, 2025],
            "source_priority": [1, 1],
            "fallback_used": [False, False],
            "candidate_source_count": [1, 1],
            "candidate_sources": ["sec_companyfacts", "sec_filing"],
            "candidate_source_labels": ["NetIncomeLoss", "EntityCommonStockSharesOutstanding"],
        }
    )

    consolidated, lineage, _ = build_sec_only_earnings(
        sec_calendar=sec_calendar,
        sec_actuals=sec_actuals,
        sec_financials=sec_financials,
    )

    assert consolidated["epsActual"].to_list() == [2.0]
    assert consolidated["actual_source"].to_list() == ["sec_derived_eps"]
    assert consolidated["selected_source"].to_list() == ["sec_submissions+sec_derived_eps"]
    assert lineage["sec_epsActual"].to_list() == [2.0]


def test_build_sec_only_earnings_uses_weighted_average_diluted_shares_when_outstanding_missing() -> None:
    sec_calendar = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "period_end": ["2025-03-31"],
            "reportDate": ["2025-05-01"],
            "earningsDatetime": [None],
            "accession_number": ["0000001"],
            "form": ["10-Q"],
            "fiscal_period": ["Q1"],
            "fiscal_year": [2025],
            "source": ["sec_submissions"],
            "source_label": ["reportDate"],
        }
    )
    sec_actuals = pl.DataFrame(
        {
            "ticker": [],
            "period_end": [],
            "reportDate": [],
            "epsActual": [],
            "source": [],
            "source_label": [],
            "form": [],
            "fiscal_period": [],
            "fiscal_year": [],
        },
        schema={
            "ticker": pl.String,
            "period_end": pl.String,
            "reportDate": pl.String,
            "epsActual": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
        },
    )
    sec_financials = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "statement": ["income_statement", "shares"],
            "metric": ["net_income", "weighted_average_diluted_shares"],
            "date": ["2025-03-31", "2025-03-31"],
            "filing_date": ["2025-05-01", "2025-05-01"],
            "value": [300.0, 150.0],
            "source": ["sec_companyfacts", "sec_companyfacts"],
            "source_label": ["NetIncomeLoss", "WeightedAverageNumberOfDilutedSharesOutstanding"],
            "selected_source": ["sec_companyfacts", "sec_companyfacts"],
            "selected_source_label": ["NetIncomeLoss", "WeightedAverageNumberOfDilutedSharesOutstanding"],
            "selected_accession_number": [None, None],
            "selected_form": ["10-Q", "10-Q"],
            "selected_fiscal_period": ["Q1", "Q1"],
            "selected_fiscal_year": [2025, 2025],
            "source_priority": [1, 1],
            "fallback_used": [False, False],
            "candidate_source_count": [1, 1],
            "candidate_sources": ["sec_companyfacts", "sec_companyfacts"],
            "candidate_source_labels": ["NetIncomeLoss", "WeightedAverageNumberOfDilutedSharesOutstanding"],
        }
    )

    consolidated, lineage, _ = build_sec_only_earnings(
        sec_calendar=sec_calendar,
        sec_actuals=sec_actuals,
        sec_financials=sec_financials,
    )

    assert consolidated["epsActual"].to_list() == [2.0]
    assert consolidated["actual_source"].to_list() == ["sec_derived_eps"]
    assert "weighted_average_diluted_shares" not in lineage.columns or lineage["sec_epsActual"].to_list() == [2.0]


def test_build_sec_only_earnings_appends_derived_eps_without_calendar_row() -> None:
    sec_calendar = pl.DataFrame(
        {
            "ticker": [],
            "period_end": [],
            "reportDate": [],
            "earningsDatetime": [],
            "accession_number": [],
            "form": [],
            "fiscal_period": [],
            "fiscal_year": [],
            "source": [],
            "source_label": [],
        },
        schema={
            "ticker": pl.String,
            "period_end": pl.String,
            "reportDate": pl.String,
            "earningsDatetime": pl.String,
            "accession_number": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
            "source": pl.String,
            "source_label": pl.String,
        },
    )
    sec_actuals = pl.DataFrame(
        {
            "ticker": [],
            "period_end": [],
            "reportDate": [],
            "epsActual": [],
            "source": [],
            "source_label": [],
            "form": [],
            "fiscal_period": [],
            "fiscal_year": [],
        },
        schema={
            "ticker": pl.String,
            "period_end": pl.String,
            "reportDate": pl.String,
            "epsActual": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
        },
    )
    sec_financials = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "statement": ["income_statement", "shares"],
            "metric": ["net_income", "outstanding_shares"],
            "date": ["2025-03-31", "2025-03-31"],
            "filing_date": ["2025-05-01", "2025-05-01"],
            "value": [200.0, 100.0],
            "source": ["sec_companyfacts", "sec_filing"],
            "source_label": ["NetIncomeLoss", "EntityCommonStockSharesOutstanding"],
            "selected_source": ["sec_companyfacts", "sec_filing"],
            "selected_source_label": ["NetIncomeLoss", "EntityCommonStockSharesOutstanding"],
            "selected_accession_number": [None, "0001"],
            "selected_form": ["10-Q", "10-Q"],
            "selected_fiscal_period": ["Q1", "Q1"],
            "selected_fiscal_year": [2025, 2025],
            "source_priority": [1, 1],
            "fallback_used": [False, False],
            "candidate_source_count": [1, 1],
            "candidate_sources": ["sec_companyfacts", "sec_filing"],
            "candidate_source_labels": ["NetIncomeLoss", "EntityCommonStockSharesOutstanding"],
        }
    )

    consolidated, lineage, long_frame = build_sec_only_earnings(
        sec_calendar=sec_calendar,
        sec_actuals=sec_actuals,
        sec_financials=sec_financials,
    )

    assert consolidated["epsActual"].to_list() == [2.0]
    assert consolidated["selected_source"].to_list() == ["sec_derived_eps_only"]
    assert lineage["actual_source"].to_list() == ["sec_derived_eps"]
    assert long_frame["metric"].to_list() == ["eps_actual"]


def test_build_sec_only_earnings_aligns_quarter_labels_to_financials_on_same_period_end() -> None:
    sec_calendar = pl.DataFrame(
        {
            "ticker": ["WSM.US"],
            "period_end": ["2024-04-28"],
            "reportDate": ["2024-05-24"],
            "earningsDatetime": [None],
            "accession_number": ["0000001"],
            "form": ["10-Q"],
            "fiscal_period": ["Q2"],
            "fiscal_year": [2024],
            "source": ["sec_submissions"],
            "source_label": ["reportDate"],
        }
    )
    sec_actuals = pl.DataFrame(
        {
            "ticker": ["WSM.US"],
            "period_end": ["2024-04-28"],
            "reportDate": ["2024-05-24"],
            "epsActual": [4.07],
            "source": ["sec_companyfacts"],
            "source_label": ["EarningsPerShareDiluted"],
            "form": ["10-Q"],
            "fiscal_period": ["Q2"],
            "fiscal_year": [2024],
        }
    )
    sec_financials = pl.DataFrame(
        {
            "ticker": ["WSM.US", "WSM.US"],
            "statement": ["income_statement", "income_statement"],
            "metric": ["revenue", "net_income"],
            "date": ["2024-04-28", "2024-04-28"],
            "filing_date": ["2024-05-24", "2024-05-24"],
            "value": [1660.3, 265.7],
            "source": ["sec_companyfacts", "sec_companyfacts"],
            "source_label": ["Revenues", "NetIncomeLoss"],
            "selected_source": ["sec_companyfacts", "sec_companyfacts"],
            "selected_source_label": ["Revenues", "NetIncomeLoss"],
            "selected_accession_number": [None, None],
            "selected_form": ["10-Q", "10-Q"],
            "selected_fiscal_period": ["Q1", "Q1"],
            "selected_fiscal_year": [2024, 2024],
            "source_priority": [1, 1],
            "fallback_used": [False, False],
            "candidate_source_count": [1, 1],
            "candidate_sources": ["sec_companyfacts", "sec_companyfacts"],
            "candidate_source_labels": ["Revenues", "NetIncomeLoss"],
        }
    )

    consolidated, lineage, long_frame = build_sec_only_earnings(
        sec_calendar=sec_calendar,
        sec_actuals=sec_actuals,
        sec_financials=sec_financials,
    )

    assert consolidated["fiscal_period"].to_list() == ["Q1"]
    assert consolidated["fiscal_year"].to_list() == [2024]
    assert lineage["fiscal_period"].to_list() == ["Q1"]
    assert not long_frame.is_empty()


def test_build_sec_only_earnings_realigns_sec_actuals_to_calendar_by_report_date() -> None:
    sec_calendar = pl.DataFrame(
        {
            "ticker": ["V.US"],
            "period_end": ["2023-09-30"],
            "reportDate": ["2023-11-15"],
            "earningsDatetime": [None],
            "accession_number": ["0001403161-23-000099"],
            "form": ["10-Q"],
            "fiscal_period": ["Q3"],
            "fiscal_year": [2023],
            "source": ["sec_submissions"],
            "source_label": ["reportDate"],
        }
    )
    sec_actuals = pl.DataFrame(
        {
            "ticker": ["V.US"],
            "period_end": ["2021-09-30"],
            "reportDate": ["2023-11-15"],
            "epsActual": [-0.39],
            "source": ["sec_filing"],
            "source_label": ["EarningsPerShareDiluted_derived"],
            "form": ["10-Q"],
            "fiscal_period": ["Q4"],
            "fiscal_year": [2023],
        }
    )

    consolidated, lineage, _ = build_sec_only_earnings(
        sec_calendar=sec_calendar,
        sec_actuals=sec_actuals,
    )

    assert consolidated["period_end"].to_list() == ["2023-09-30"]
    assert consolidated["epsActual"].to_list() == [-0.39]
    assert consolidated["actual_source"].to_list() == ["sec_filing"]
    assert lineage["sec_reportDate"].to_list() == ["2023-11-15"]
    assert lineage["sec_epsActual"].to_list() == [-0.39]


def test_build_sec_only_earnings_derived_eps_prefers_closest_share_base_for_same_quarter() -> None:
    sec_calendar = pl.DataFrame(
        {
            "ticker": ["KO.US"],
            "period_end": ["2021-12-31"],
            "reportDate": ["2022-02-22"],
            "earningsDatetime": [None],
            "accession_number": ["0001"],
            "form": ["10-K"],
            "fiscal_period": ["Q4"],
            "fiscal_year": [2021],
            "source": ["sec_submissions"],
            "source_label": ["reportDate"],
        }
    )
    sec_actuals = pl.DataFrame(
        schema={
            "ticker": pl.String,
            "period_end": pl.String,
            "reportDate": pl.String,
            "epsActual": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
        }
    )
    sec_financials = pl.DataFrame(
        {
            "ticker": ["KO.US", "KO.US", "KO.US"],
            "statement": ["income_statement", "shares", "shares"],
            "metric": ["net_income", "outstanding_shares", "weighted_average_diluted_shares"],
            "date": ["2021-12-31", "2021-12-31", "2021-10-01"],
            "filing_date": ["2022-02-20", "2022-02-22", "2022-10-28"],
            "value": [1_030_000_000.0, 4_325_000_000.0, 4_000_000_000.0],
            "source": ["sec_companyfacts", "sec_filing", "sec_companyfacts"],
            "source_label": ["NetIncomeLoss", "CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
            "selected_source": ["sec_companyfacts", "sec_filing", "sec_companyfacts"],
            "selected_source_label": ["NetIncomeLoss", "CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
            "selected_accession_number": [None, "0001", None],
            "selected_form": ["10-K", "10-K", "10-Q"],
            "selected_fiscal_period": ["Q4", "Q4", "Q4"],
            "selected_fiscal_year": [2021, 2021, 2021],
            "source_priority": [1, 1, 1],
            "fallback_used": [False, False, False],
            "candidate_source_count": [1, 1, 1],
            "candidate_sources": ["sec_companyfacts", "sec_filing", "sec_companyfacts"],
            "candidate_source_labels": [
                "NetIncomeLoss",
                "CommonStockSharesOutstanding",
                "WeightedAverageNumberOfDilutedSharesOutstanding",
            ],
        }
    )

    consolidated, lineage, _ = build_sec_only_earnings(
        sec_calendar=sec_calendar,
        sec_actuals=sec_actuals,
        sec_financials=sec_financials,
    )

    assert consolidated["epsActual"].to_list() == [1_030_000_000.0 / 4_325_000_000.0]
    assert consolidated["actual_source"].to_list() == ["sec_derived_eps"]
    assert lineage["sec_epsActual"].to_list() == [1_030_000_000.0 / 4_325_000_000.0]


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


def test_build_sec_only_financials_preserves_valid_off_calendar_fiscal_years() -> None:
    sec_companyfacts = pl.DataFrame(
        {
            "ticker": ["SWY.US", "SWY.US", "SWY.US", "SWY.US", "SWY.US", "SWY.US"],
            "statement": ["income_statement"] * 6,
            "metric": ["revenue", "revenue", "revenue", "net_income", "net_income", "net_income"],
            "date": ["2014-03-22", "2014-06-14", "2015-01-03", "2014-03-22", "2014-06-14", "2015-01-03"],
            "filing_date": ["2014-05-06", "2014-07-27", "2015-03-04", "2014-05-06", "2014-07-27", "2015-03-04"],
            "value": [8260.0, 8307.0, 11677.0, 193.0, 234.0, 106.0],
            "source": ["sec_companyfacts"] * 6,
            "source_label": ["SalesRevenueNet", "SalesRevenueNet", "SalesRevenueNet", "NetIncomeLoss", "NetIncomeLoss", "NetIncomeLoss"],
            "accession_number": [None] * 6,
            "form": ["10-Q", "10-Q", "10-K", "10-Q", "10-Q", "10-K"],
            "fiscal_period": ["Q1", "Q2", "Q4", "Q1", "Q2", "Q4"],
            "fiscal_year": [2014, 2014, 2014, 2014, 2014, 2014],
        }
    )
    sec_filing = sec_companyfacts.head(0)

    consolidated, _, _ = build_sec_only_financials(
        sec_companyfacts=sec_companyfacts,
        sec_filing=sec_filing,
    )

    swy_q4 = consolidated.filter(
        (pl.col("ticker") == "SWY.US")
        & (pl.col("date") == "2015-01-03")
        & (pl.col("metric") == "revenue")
    )
    assert swy_q4["selected_fiscal_year"].to_list() == [2014]
    assert swy_q4["selected_fiscal_period"].to_list() == ["Q4"]


def test_build_sec_only_financials_uses_august_year_end_for_costco_style_fiscal_years() -> None:
    sec_companyfacts = pl.DataFrame(
        {
            "ticker": ["COST.US"] * 4,
            "statement": ["income_statement"] * 4,
            "metric": ["revenue"] * 4,
            "date": ["2021-11-21", "2022-02-13", "2022-05-08", "2022-08-28"],
            "filing_date": ["2021-12-09", "2022-03-10", "2022-05-26", "2022-09-22"],
            "value": [50.0, 51.0, 52.0, 53.0],
            "source": ["sec_companyfacts"] * 4,
            "source_label": ["RevenueFromContractWithCustomerExcludingAssessedTax"] * 4,
            "accession_number": [None] * 4,
            "form": ["10-Q", "10-Q", "10-Q", "10-K"],
            "fiscal_period": ["Q1", "Q2", "Q3", "Q4"],
            "fiscal_year": [2022, 2022, 2022, 2024],
        }
    )
    sec_filing = sec_companyfacts.head(0)

    consolidated, _, _ = build_sec_only_financials(
        sec_companyfacts=sec_companyfacts,
        sec_filing=sec_filing,
    )

    rows = consolidated.filter(pl.col("ticker") == "COST.US").sort("date")
    assert rows["selected_fiscal_year"].to_list() == [2022, 2022, 2022, 2022]
    assert rows["selected_fiscal_period"].to_list() == ["Q1", "Q2", "Q3", "Q4"]


def test_build_sec_only_financials_uses_end_year_for_february_year_end_sequences() -> None:
    sec_companyfacts = pl.DataFrame(
        {
            "ticker": ["STZ.US"] * 5,
            "statement": ["income_statement"] * 5,
            "metric": ["revenue"] * 5,
            "date": ["2016-05-31", "2016-08-31", "2016-11-30", "2016-11-30", "2017-02-28"],
            "filing_date": ["2016-06-30", "2016-10-05", "2017-01-05", "2017-01-05", "2017-04-20"],
            "value": [2000.0, 2100.0, 2200.0, 2200.0, 2300.0],
            "source": ["sec_companyfacts"] * 5,
            "source_label": ["RevenueFromContractWithCustomerExcludingAssessedTax"] * 5,
            "accession_number": [None] * 5,
            "form": ["10-Q", "10-Q", "10-Q", "10-Q", "10-K"],
            "fiscal_period": ["Q1", "Q2", "Q3", "Q4", "Q1"],
            "fiscal_year": [2016, 2016, 2016, 2016, 2017],
        }
    )

    consolidated, _, _ = build_sec_only_financials(
        sec_companyfacts=sec_companyfacts,
        sec_filing=sec_companyfacts.head(0),
    )

    rows = (
        consolidated.filter((pl.col("ticker") == "STZ.US") & (pl.col("metric") == "revenue"))
        .sort("date")
        .select(["date", "selected_fiscal_year", "selected_fiscal_period"])
    )
    assert rows["date"].to_list() == ["2016-05-31", "2016-08-31", "2016-11-30", "2017-02-28"]
    assert rows["selected_fiscal_year"].to_list() == [2017, 2017, 2017, 2017]
    assert rows["selected_fiscal_period"].to_list() == ["Q1", "Q2", "Q3", "Q4"]


def test_build_sec_only_financials_uses_month_sequence_for_january_year_end_quarters() -> None:
    sec_companyfacts = pl.DataFrame(
        {
            "ticker": ["AAP.US"] * 5,
            "statement": ["income_statement"] * 5,
            "metric": ["revenue"] * 5,
            "date": ["2024-01-13", "2024-04-20", "2024-07-13", "2024-10-05", "2024-12-28"],
            "filing_date": ["2024-02-28", "2024-05-30", "2024-08-22", "2024-11-14", "2025-02-26"],
            "value": [2100.0, 3400.0, 2680.0, 2148.0, 2148.0],
            "source": ["sec_companyfacts"] * 5,
            "source_label": ["RevenueFromContractWithCustomerExcludingAssessedTax"] * 5,
            "accession_number": [None] * 5,
            "form": ["10-K", "10-Q", "10-Q", "10-Q", "10-K"],
            "fiscal_period": ["Q4", "Q1", "Q2", "Q3", "Q4"],
            "fiscal_year": [2023, 2024, 2024, 2024, 2024],
        }
    )

    consolidated, _, _ = build_sec_only_financials(
        sec_companyfacts=sec_companyfacts,
        sec_filing=sec_companyfacts.head(0),
    )

    rows = (
        consolidated.filter((pl.col("ticker") == "AAP.US") & (pl.col("metric") == "revenue"))
        .sort("date")
        .select(["date", "selected_fiscal_year", "selected_fiscal_period"])
    )
    assert rows["selected_fiscal_year"].to_list() == [2023, 2024, 2024, 2024, 2024]
    assert rows["selected_fiscal_period"].to_list() == ["Q4", "Q1", "Q2", "Q3", "Q4"]


def test_build_sec_only_financials_preserves_reliable_source_quarters_when_q4_is_missing() -> None:
    sec_companyfacts = pl.DataFrame(
        {
            "ticker": ["AAP.US"] * 4,
            "statement": ["income_statement"] * 4,
            "metric": ["revenue"] * 4,
            "date": ["2024-04-20", "2024-07-13", "2024-10-05", "2025-04-19"],
            "filing_date": ["2024-05-30", "2024-08-22", "2024-11-14", "2025-05-22"],
            "value": [3400.0, 2680.0, 2148.0, 2583.0],
            "source": ["sec_companyfacts"] * 4,
            "source_label": ["RevenueFromContractWithCustomerExcludingAssessedTax"] * 4,
            "accession_number": [None] * 4,
            "form": ["10-Q", "10-Q", "10-Q", "10-Q"],
            "fiscal_period": ["Q1", "Q2", "Q3", "Q1"],
            "fiscal_year": [2024, 2024, 2024, 2025],
        }
    )

    consolidated, _, _ = build_sec_only_financials(
        sec_companyfacts=sec_companyfacts,
        sec_filing=sec_companyfacts.head(0),
    )

    rows = (
        consolidated.filter((pl.col("ticker") == "AAP.US") & (pl.col("metric") == "revenue"))
        .sort("date")
        .select(["date", "selected_fiscal_year", "selected_fiscal_period"])
    )
    assert rows["selected_fiscal_year"].to_list() == [2024, 2024, 2024, 2025]
    assert rows["selected_fiscal_period"].to_list() == ["Q1", "Q2", "Q3", "Q1"]


def test_build_sec_only_financials_preserves_recent_locally_coherent_source_quarters_even_if_old_history_is_noisy() -> None:
    sec_companyfacts = pl.DataFrame(
        {
            "ticker": ["AAP.US"] * 8,
            "statement": ["income_statement"] * 8,
            "metric": ["revenue"] * 8,
            "date": [
                "2010-01-02",
                "2010-04-24",
                "2010-07-17",
                "2010-10-09",
                "2024-04-20",
                "2024-07-13",
                "2024-10-05",
                "2025-04-19",
            ],
            "filing_date": [
                "2010-03-02",
                "2010-05-26",
                "2010-08-24",
                "2010-11-16",
                "2024-05-30",
                "2024-08-22",
                "2024-11-14",
                "2025-05-22",
            ],
            "value": [2000.0, 2100.0, 2200.0, 2300.0, 3400.0, 2680.0, 2148.0, 2583.0],
            "source": ["sec_companyfacts"] * 8,
            "source_label": ["RevenueFromContractWithCustomerExcludingAssessedTax"] * 8,
            "accession_number": [None] * 8,
            "form": ["10-K", "10-Q", "10-Q", "10-Q", "10-Q", "10-Q", "10-Q", "10-Q"],
            "fiscal_period": ["Q4", "Q2", "Q3", "Q3", "Q1", "Q2", "Q3", "Q1"],
            "fiscal_year": [2010, 2010, 2010, 2010, 2024, 2024, 2024, 2025],
        }
    )

    consolidated, _, _ = build_sec_only_financials(
        sec_companyfacts=sec_companyfacts,
        sec_filing=sec_companyfacts.head(0),
    )

    recent_rows = (
        consolidated.filter((pl.col("ticker") == "AAP.US") & (pl.col("date") >= "2024-04-20"))
        .sort("date")
        .select(["date", "selected_fiscal_year", "selected_fiscal_period"])
    )
    assert recent_rows["selected_fiscal_year"].to_list() == [2024, 2024, 2024, 2025]
    assert recent_rows["selected_fiscal_period"].to_list() == ["Q1", "Q2", "Q3", "Q1"]


def test_build_quarterly_presence_uses_separate_financial_and_eps_grids() -> None:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "open_source" / "build_sec_quality_dashboard.py"
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
