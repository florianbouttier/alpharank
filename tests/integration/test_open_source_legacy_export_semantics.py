from __future__ import annotations

from pathlib import Path

import polars as pl
from _legacy_export_support import (
    _write_minimal_legacy_reference,
)

from alpharank.data.open_source.legacy_export import export_legacy_compatible_outputs


def test_export_legacy_compatible_outputs_prefers_vendor_when_q4_sec_outlier_breaks_series(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "live" / "legacy"
    reference_dir.mkdir(parents=True)
    _write_minimal_legacy_reference(reference_dir, ticker="KEYS.US", code="KEYS", name="Keysight")

    clean_prices = pl.read_parquet(reference_dir / "US_Finalprice.parquet")
    benchmark_prices = pl.read_parquet(reference_dir / "SP500Price.parquet")
    general_reference = pl.DataFrame(
        {
            "ticker": ["KEYS.US"],
            "name": ["Keysight"],
            "exchange": ["NYSE"],
            "cik": ["0001601046"],
            "source": ["open_source_general"],
            "Sector": ["Technology"],
            "industry": ["Electronic Equipment"],
            "sector_source": ["yfinance"],
            "sector_raw_value": ["Technology"],
            "sic": [None],
            "sic_description": [None],
            "mapping_rule": ["yfinance:sector"],
        }
    )
    empty_consolidated = pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "filing_date": pl.String,
            "value": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "selected_source": pl.String,
            "selected_source_label": pl.String,
            "selected_form": pl.String,
            "selected_fiscal_period": pl.String,
            "selected_fiscal_year": pl.Int64,
            "source_priority": pl.Int64,
        }
    )
    consolidated_lineage = pl.DataFrame(
        {
            "ticker": ["KEYS.US"] * 4,
            "statement": ["income_statement"] * 4,
            "metric": ["revenue"] * 4,
            "date": ["2025-07-31", "2025-10-31", "2025-10-31", "2026-01-31"],
            "filing_date": ["2025-08-29", "2025-12-17", None, "2026-03-05"],
            "value": [1_352_000_000.0, 107_000_000.0, 1_419_000_000.0, 1_600_000_000.0],
            "source": ["sec_companyfacts", "sec_filing", "yfinance", "sec_companyfacts"],
            "source_label": ["tag", "xbrl", "statement row", "tag"],
            "selected_source": ["sec_companyfacts", "sec_filing", "yfinance", "sec_companyfacts"],
            "selected_source_label": ["tag", "xbrl", "statement row", "tag"],
            "selected_form": ["10-Q", "10-K", None, "10-Q"],
            "selected_fiscal_period": ["Q3", "Q4", None, "Q1"],
            "selected_fiscal_year": [2025, 2025, None, 2026],
            "source_priority": [1, 2, 4, 1],
        }
    )

    export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=benchmark_prices,
        general_reference=general_reference,
        consolidated_financials=empty_consolidated,
        consolidated_lineage=consolidated_lineage,
        earnings_frame=pl.DataFrame(),
        reference_data_dir=reference_dir,
        output_dir=output_dir,
    )

    income = pl.read_parquet(output_dir / "US_Income_statement.parquet").sort("date")
    assert income.filter(pl.col("date") == "2025-12-31")["totalRevenue"].to_list() == [1_419_000_000.0]
    assert income.filter(pl.col("date") == "2025-12-31")["filing_date"].to_list() == ["2025-12-17"]


def test_export_legacy_compatible_outputs_prefers_vendor_revenue_for_reits_and_financials(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "live" / "legacy"
    reference_dir.mkdir(parents=True)
    _write_minimal_legacy_reference(reference_dir, ticker="AVB.US", code="AVB", name="AvalonBay")

    clean_prices = pl.read_parquet(reference_dir / "US_Finalprice.parquet")
    benchmark_prices = pl.read_parquet(reference_dir / "SP500Price.parquet")
    general_reference = pl.DataFrame(
        {
            "ticker": ["AVB.US", "HIG.US", "BG.US"],
            "name": ["AvalonBay", "Hartford", "Bunge"],
            "exchange": ["NYSE", "NYSE", "NYSE"],
            "cik": ["0000915912", "0000874766", "0000014681"],
            "source": ["open_source_general", "open_source_general", "open_source_general"],
            "Sector": ["Real Estate", "Financial Services", "Consumer Defensive"],
            "industry": ["REIT - Residential", "Insurance - Diversified", "Farm Products"],
            "sector_source": ["yfinance", "yfinance", "yfinance"],
            "sector_raw_value": ["Real Estate", "Financial Services", "Consumer Defensive"],
            "sic": [None, None, None],
            "sic_description": [None, None, None],
            "mapping_rule": ["yfinance:sector", "yfinance:sector", "yfinance:sector"],
        }
    )
    empty_consolidated = pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "filing_date": pl.String,
            "value": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "selected_source": pl.String,
            "selected_source_label": pl.String,
            "selected_form": pl.String,
            "selected_fiscal_period": pl.String,
            "selected_fiscal_year": pl.Int64,
            "source_priority": pl.Int64,
        }
    )
    consolidated_lineage = pl.DataFrame(
        {
            "ticker": ["AVB.US", "AVB.US", "AVB.US", "HIG.US", "HIG.US", "HIG.US", "BG.US", "BG.US", "BG.US"],
            "statement": ["income_statement"] * 9,
            "metric": ["revenue"] * 9,
            "date": ["2025-03-31"] * 9,
            "filing_date": ["2025-05-08"] * 3 + ["2025-04-24"] * 3 + ["2025-05-07"] * 3,
            "value": [
                1_742_000.0,
                1_742_000.0,
                745_880_000.0,
                366_000_000.0,
                366_000_000.0,
                6_810_000_000.0,
                3_663_000_000.0,
                3_663_000_000.0,
                11_643_000_000.0,
            ],
            "source": ["sec_companyfacts", "sec_filing", "yfinance", "sec_companyfacts", "sec_filing", "yfinance", "sec_companyfacts", "sec_filing", "yfinance"],
            "source_label": ["tag", "xbrl", "statement row", "tag", "xbrl", "statement row", "tag", "xbrl", "statement row"],
            "selected_source": ["sec_companyfacts", "sec_filing", "yfinance", "sec_companyfacts", "sec_filing", "yfinance", "sec_companyfacts", "sec_filing", "yfinance"],
            "selected_source_label": ["tag", "xbrl", "statement row", "tag", "xbrl", "statement row", "tag", "xbrl", "statement row"],
            "selected_form": ["10-Q", "10-Q", None, "10-Q", "10-Q", None, "10-Q", "10-Q", None],
            "selected_fiscal_period": ["Q1", "Q1", None, "Q1", "Q1", None, "Q1", "Q1", None],
            "selected_fiscal_year": [2025, 2025, None, 2025, 2025, None, 2025, 2025, None],
            "source_priority": [1, 2, 4, 1, 2, 4, 1, 2, 4],
        }
    )

    export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=benchmark_prices,
        general_reference=general_reference,
        consolidated_financials=empty_consolidated,
        consolidated_lineage=consolidated_lineage,
        earnings_frame=pl.DataFrame(),
        reference_data_dir=reference_dir,
        output_dir=output_dir,
    )

    income = pl.read_parquet(output_dir / "US_Income_statement.parquet").sort(["ticker", "date"])
    assert income.filter(pl.col("ticker") == "AVB.US")["totalRevenue"].to_list() == [745_880_000.0]
    assert income.filter(pl.col("ticker") == "HIG.US")["totalRevenue"].to_list() == [6_810_000_000.0]
    assert income.filter(pl.col("ticker") == "BG.US")["totalRevenue"].to_list() == [11_643_000_000.0]


def test_export_legacy_compatible_outputs_keeps_distinct_dates_even_when_filing_date_matches(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "live" / "legacy"
    reference_dir.mkdir(parents=True)
    _write_minimal_legacy_reference(reference_dir, ticker="JEF.US", code="JEF", name="Jefferies")
    pl.DataFrame(
        {
            "ticker": ["JEF.US"],
            "date": ["2025-05-31"],
            "filing_date": ["2025-07-09"],
            "commonStockSharesOutstanding": ["0.0"],
            "totalLiab": [0.0],
            "totalStockholderEquity": [0.0],
        }
    ).write_parquet(reference_dir / "US_Balance_sheet.parquet")

    clean_prices = pl.read_parquet(reference_dir / "US_Finalprice.parquet")
    benchmark_prices = pl.read_parquet(reference_dir / "SP500Price.parquet")
    general_reference = pl.DataFrame(
        {
            "ticker": ["JEF.US"],
            "name": ["Jefferies"],
            "exchange": ["NYSE"],
            "cik": ["0000964130"],
            "source": ["open_source_general"],
            "Sector": ["Financial Services"],
            "industry": ["Capital Markets"],
            "sector_source": ["yfinance"],
            "sector_raw_value": ["Financial Services"],
            "sic": [None],
            "sic_description": [None],
            "mapping_rule": ["yfinance:sector"],
        }
    )
    empty_consolidated = pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "filing_date": pl.String,
            "value": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "selected_source": pl.String,
            "selected_source_label": pl.String,
            "selected_form": pl.String,
            "selected_fiscal_period": pl.String,
            "selected_fiscal_year": pl.Int64,
            "source_priority": pl.Int64,
        }
    )
    consolidated_lineage = pl.DataFrame(
        {
            "ticker": ["JEF.US"] * 6,
            "statement": ["balance_sheet"] * 6,
            "metric": ["total_liabilities", "total_liabilities", "total_liabilities", "stockholders_equity", "stockholders_equity", "stockholders_equity"],
            "date": ["2025-03-31", "2025-05-31", "2025-05-31", "2025-03-31", "2025-05-31", "2025-05-31"],
            "filing_date": ["2025-07-09"] * 6,
            "value": [100_000.0, 56_902_764_000.0, 56_902_764_000.0, 447_800_000.0, 10_305_025_000.0, 10_305_025_000.0],
            "source": ["sec_filing", "sec_companyfacts", "yfinance", "sec_filing", "sec_companyfacts", "yfinance"],
            "source_label": ["xbrl", "tag", "statement row", "xbrl", "tag", "statement row"],
            "selected_source": ["sec_filing", "sec_companyfacts", "yfinance", "sec_filing", "sec_companyfacts", "yfinance"],
            "selected_source_label": ["xbrl", "tag", "statement row", "xbrl", "tag", "statement row"],
            "selected_form": ["10-Q", "10-Q", None, "10-Q", "10-Q", None],
            "selected_fiscal_period": ["Q2", "Q2", None, "Q2", "Q2", None],
            "selected_fiscal_year": [2025, 2025, None, 2025, 2025, None],
            "source_priority": [2, 1, 4, 2, 1, 4],
        }
    )

    export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=benchmark_prices,
        general_reference=general_reference,
        consolidated_financials=empty_consolidated,
        consolidated_lineage=consolidated_lineage,
        earnings_frame=pl.DataFrame(),
        reference_data_dir=reference_dir,
        output_dir=output_dir,
    )

    balance = pl.read_parquet(output_dir / "US_Balance_sheet.parquet").sort("date")
    assert balance["date"].to_list() == ["2025-03-31", "2025-06-30"]
    assert balance["totalLiab"].to_list() == [100_000.0, 56_902_764_000.0]
    assert balance["totalStockholderEquity"].to_list() == [447_800_000.0, 10_305_025_000.0]


def test_export_legacy_compatible_outputs_fills_or_drops_null_filing_dates(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "live" / "legacy"
    reference_dir.mkdir(parents=True)
    _write_minimal_legacy_reference(reference_dir, ticker="ACN.US", code="ACN", name="Accenture")

    clean_prices = pl.read_parquet(reference_dir / "US_Finalprice.parquet")
    benchmark_prices = pl.read_parquet(reference_dir / "SP500Price.parquet")
    general_reference = pl.DataFrame(
        {
            "ticker": ["ACN.US"],
            "name": ["Accenture"],
            "exchange": ["NYSE"],
            "cik": ["0001467373"],
            "source": ["open_source_general"],
            "Sector": ["Technology"],
            "industry": ["IT Services"],
            "sector_source": ["yfinance"],
            "sector_raw_value": ["Technology"],
            "sic": [None],
            "sic_description": [None],
            "mapping_rule": ["yfinance:sector"],
        }
    )
    empty_consolidated = pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "filing_date": pl.String,
            "value": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "selected_source": pl.String,
            "selected_source_label": pl.String,
            "selected_form": pl.String,
            "selected_fiscal_period": pl.String,
            "selected_fiscal_year": pl.Int64,
            "source_priority": pl.Int64,
        }
    )
    consolidated_lineage = pl.DataFrame(
        {
            "ticker": ["ACN.US", "ACN.US", "ACN.US", "ACN.US", "ACN.US"],
            "statement": ["income_statement", "income_statement", "balance_sheet", "cash_flow", "cash_flow"],
            "metric": ["revenue", "net_income", "total_assets", "free_cash_flow", "free_cash_flow"],
            "date": ["2025-02-28", "2025-02-28", "2025-02-28", "2025-02-28", "2026-02-28"],
            "filing_date": ["2025-03-20", "2025-03-20", "2025-03-20", None, None],
            "value": [16_659_301_000.0, 1_788_075_000.0, 29_246_053_000.0, 2_682_588_000.0, 3_667_953_000.0],
            "source": ["yfinance", "yfinance", "yfinance", "yfinance", "yfinance"],
            "source_label": ["Total Revenue", "Net Income", "Total Assets", "Free Cash Flow", "Free Cash Flow"],
            "selected_source": ["yfinance"] * 5,
            "selected_source_label": ["statement row"] * 5,
            "selected_form": [None] * 5,
            "selected_fiscal_period": [None] * 5,
            "selected_fiscal_year": [None] * 5,
            "source_priority": [4] * 5,
        }
    )

    export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=benchmark_prices,
        general_reference=general_reference,
        consolidated_financials=empty_consolidated,
        consolidated_lineage=consolidated_lineage,
        earnings_frame=pl.DataFrame(),
        reference_data_dir=reference_dir,
        output_dir=output_dir,
    )

    cash = pl.read_parquet(output_dir / "US_Cash_flow.parquet").sort("date")
    assert cash["date"].to_list() == ["2025-03-31"]
    assert cash["filing_date"].to_list() == ["2025-03-20"]
    assert cash["freeCashFlow"].to_list() == [2_682_588_000.0]


def test_export_legacy_compatible_outputs_keeps_distinct_quarters_from_same_filing_date(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "live" / "legacy"
    reference_dir.mkdir(parents=True)
    _write_minimal_legacy_reference(reference_dir, ticker="AAPL.US", code="AAPL", name="Apple")

    clean_prices = pl.read_parquet(reference_dir / "US_Finalprice.parquet")
    benchmark_prices = pl.read_parquet(reference_dir / "SP500Price.parquet")
    general_reference = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "name": ["Apple"],
            "exchange": ["NASDAQ"],
            "cik": ["0000320193"],
            "source": ["sec_mapping"],
            "Sector": ["Technology"],
            "industry": ["Consumer Electronics"],
            "sector_source": ["sec_sic"],
            "sector_raw_value": ["Technology"],
            "sic": [None],
            "sic_description": [None],
            "mapping_rule": ["sec"],
        }
    )
    empty_consolidated = pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "filing_date": pl.String,
            "value": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "selected_source": pl.String,
            "selected_source_label": pl.String,
            "selected_form": pl.String,
            "selected_fiscal_period": pl.String,
            "selected_fiscal_year": pl.Int64,
            "source_priority": pl.Int64,
        }
    )
    consolidated_lineage = pl.DataFrame(
        {
            "ticker": ["AAPL.US"] * 4,
            "statement": ["income_statement", "income_statement", "shares", "shares"],
            "metric": ["revenue", "revenue", "outstanding_shares", "outstanding_shares"],
            "date": ["2019-09-28", "2020-06-27", "2019-09-28", "2020-06-27"],
            "filing_date": ["2020-07-31", "2020-07-31", "2020-07-31", "2020-07-31"],
            "value": [64_040_000_000.0, 59_685_000_000.0, 4_443_236_000.0, 4_275_634_000.0],
            "source": ["sec_companyfacts"] * 4,
            "source_label": ["value"] * 4,
            "selected_source": ["sec_companyfacts"] * 4,
            "selected_source_label": ["EntityCommonStockSharesOutstanding"] * 4,
            "selected_form": ["10-Q"] * 4,
            "selected_fiscal_period": ["Q4", "Q3", "Q4", "Q3"],
            "selected_fiscal_year": [2019, 2020, 2019, 2020],
            "source_priority": [1] * 4,
        }
    )

    export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=benchmark_prices,
        general_reference=general_reference,
        consolidated_financials=empty_consolidated,
        consolidated_lineage=consolidated_lineage,
        earnings_frame=pl.DataFrame(),
        reference_data_dir=reference_dir,
        output_dir=output_dir,
    )

    income = pl.read_parquet(output_dir / "US_Income_statement.parquet").sort("date")
    shares = pl.read_parquet(output_dir / "US_share.parquet").sort("dateFormatted")

    assert income["date"].to_list() == ["2019-09-30", "2020-06-30"]
    assert income["totalRevenue"].to_list() == [64_040_000_000.0, 59_685_000_000.0]
    assert shares["dateFormatted"].to_list() == ["2019-09-30", "2020-06-30"]
    assert shares["shares"].to_list() == [4_443_236_000.0, 4_275_634_000.0]


def test_share_export_prefers_earliest_filing_within_calendarized_quarter(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "live" / "legacy"
    reference_dir.mkdir(parents=True)
    _write_minimal_legacy_reference(reference_dir, ticker="A.US", code="A", name="Agilent")
    financials = pl.DataFrame(
        {
            "ticker": ["A.US"] * 3,
            "statement": ["shares"] * 3,
            "metric": ["outstanding_shares"] * 3,
            "date": ["2011-10-31", "2011-12-01", "2011-12-31"],
            "filing_date": ["2012-06-04", "2011-12-16", "2012-03-05"],
            "value": [591_000_000.0, 348_125_175.0, 593_000_000.0],
            "source": ["sec_companyfacts"] * 3,
            "source_label": ["shares"] * 3,
            "selected_source": ["sec_companyfacts"] * 3,
            "selected_source_label": ["shares"] * 3,
            "selected_form": ["10-Q", "10-K", "10-Q"],
            "selected_fiscal_period": ["Q2", "FY", "Q4"],
            "selected_fiscal_year": [2012, 2011, 2012],
            "source_priority": [1, 1, 1],
        }
    )

    export_legacy_compatible_outputs(
        clean_prices=pl.read_parquet(reference_dir / "US_Finalprice.parquet"),
        benchmark_prices=pl.read_parquet(reference_dir / "SP500Price.parquet"),
        general_reference=pl.DataFrame(
            {"ticker": ["A.US"], "name": ["Agilent"], "exchange": ["NYSE"], "cik": ["0001090872"]}
        ),
        consolidated_financials=financials,
        consolidated_lineage=financials,
        earnings_frame=pl.DataFrame(),
        reference_data_dir=reference_dir,
        output_dir=output_dir,
    )

    shares = pl.read_parquet(output_dir / "US_share.parquet")
    assert shares.select("dateFormatted", "shares").rows() == [
        ("2011-12-31", 348_125_175.0)
    ]


def test_export_legacy_compatible_outputs_can_skip_earnings_implied_share_alignment(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "live" / "legacy"
    reference_dir.mkdir(parents=True)
    _write_minimal_legacy_reference(reference_dir, ticker="ABC.US", code="ABC", name="ABC Corp")
    pl.DataFrame(
        {
            "ticker": ["ABC.US"],
            "date": ["2025-03-31"],
            "filing_date": ["2025-05-01"],
            "totalRevenue": [100.0],
            "netIncome": [20_000_000.0],
        }
    ).write_parquet(reference_dir / "US_Income_statement.parquet")

    clean_prices = pl.read_parquet(reference_dir / "US_Finalprice.parquet")
    benchmark_prices = pl.read_parquet(reference_dir / "SP500Price.parquet")
    general_reference = pl.DataFrame(
        {
            "ticker": ["ABC.US"],
            "name": ["ABC Corp"],
            "exchange": ["NYSE"],
            "cik": ["0000000001"],
            "source": ["sec_mapping"],
            "Sector": ["Industrials"],
            "industry": ["Industrial Machinery"],
            "sector_source": ["sec_sic"],
            "sector_raw_value": ["Industrials"],
            "sic": [None],
            "sic_description": [None],
            "mapping_rule": ["sec"],
        }
    )
    consolidated_financials = pl.DataFrame(
        {
            "ticker": ["ABC.US", "ABC.US", "ABC.US"],
            "statement": ["income_statement", "balance_sheet", "shares"],
            "metric": ["net_income", "total_assets", "outstanding_shares"],
            "date": ["2025-03-31", "2025-03-31", "2025-03-31"],
            "filing_date": ["2025-05-01", "2025-05-01", "2025-05-01"],
            "value": [20_000_000.0, 500.0, 10_000_000.0],
            "source": ["open_source_consolidated"] * 3,
            "source_label": ["value"] * 3,
            "selected_source": ["sec_companyfacts"] * 3,
            "selected_source_label": ["tag"] * 3,
            "selected_accession_number": [None, None, None],
            "selected_form": ["10-Q"] * 3,
            "selected_fiscal_period": ["Q1"] * 3,
            "selected_fiscal_year": [2025] * 3,
            "source_priority": [1] * 3,
            "fallback_used": [False] * 3,
            "candidate_source_count": [1] * 3,
            "candidate_sources": ["sec_companyfacts"] * 3,
            "candidate_source_labels": ["tag"] * 3,
        }
    )
    earnings = pl.DataFrame(
        {
            "ticker": ["ABC.US"],
            "reportDate": ["2025-05-01"],
            "earningsDatetime": ["2025-05-01 20:00:00"],
            "period_end": ["2025-03-31"],
            "epsEstimate": [1.4],
            "epsActual": [2.0],
            "surprisePercent": [7.0],
            "source": ["sec_companyfacts"],
        }
    )

    export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=benchmark_prices,
        general_reference=general_reference,
        consolidated_financials=consolidated_financials,
        earnings_frame=earnings,
        reference_data_dir=reference_dir,
        output_dir=output_dir,
        align_shares_with_earnings_semantics=False,
    )

    balance = pl.read_parquet(output_dir / "US_Balance_sheet.parquet")
    assert balance["commonStockSharesOutstanding"].to_list() == [10_000_000.0]
