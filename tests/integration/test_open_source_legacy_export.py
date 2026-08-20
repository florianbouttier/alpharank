from __future__ import annotations

from pathlib import Path

import polars as pl
from _legacy_export_support import (
    _write_minimal_legacy_reference,
)

from alpharank.data.open_source.legacy_export import export_legacy_compatible_outputs


def test_export_legacy_compatible_outputs_aligns_to_reference_schemas(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "live" / "legacy"
    reference_dir.mkdir(parents=True)

    pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "date": ["2025-01-01"],
            "adjusted_close": [1.0],
            "close": [1.0],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "volume": [10.0],
        }
    ).write_parquet(reference_dir / "US_Finalprice.parquet")
    pl.DataFrame(
        {
            "ticker": ["SPY.US"],
            "date": ["2025-01-01"],
            "adjusted_close": [1.0],
            "close": [1.0],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "volume": [10.0],
        }
    ).write_parquet(reference_dir / "SP500Price.parquet")
    pl.DataFrame({"Code": ["AAPL"], "Name": ["Apple"], "Exchange": ["NASDAQ"], "CurrencyCode": ["USD"], "CurrencySymbol": ["$"], "CIK": ["0000320193"], "Sector": [""], "Industry": [""]}).write_parquet(
        reference_dir / "US_General.parquet"
    )
    pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-03-31"], "filing_date": ["2025-05-01"], "totalRevenue": [100.0], "netIncome": [20.0]}).write_parquet(
        reference_dir / "US_Income_statement.parquet"
    )
    pl.DataFrame(
        {
                "ticker": ["AAPL.US"],
                "date": ["2025-03-31"],
                "filing_date": ["2025-05-01"],
                "commonStockSharesOutstanding": ["0.0"],
                "totalAssets": [500.0],
                "totalLiab": [300.0],
            }
    ).write_parquet(
        reference_dir / "US_Balance_sheet.parquet"
    )
    pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-03-31"], "filing_date": ["2025-05-01"], "freeCashFlow": [50.0]}).write_parquet(
        reference_dir / "US_Cash_flow.parquet"
    )
    pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-03-31"], "dateFormatted": ["2025-03-31"], "sharesMln": [10.0], "shares": [10_000_000.0]}).write_parquet(
        reference_dir / "US_share.parquet"
    )
    pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "beforeAfterMarket": [""],
            "currency": [""],
            "date": ["2025-03-31"],
            "epsActual": [1.5],
            "epsDifference": [0.1],
            "epsEstimate": [1.4],
            "reportDate": ["2025-05-01"],
            "surprisePercent": [7.0],
        }
    ).write_parquet(reference_dir / "US_Earnings.parquet")

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
            "sector_source": ["yfinance"],
            "sector_raw_value": ["Technology"],
            "sic": [None],
            "sic_description": [None],
            "mapping_rule": ["yfinance:sector"],
        }
    )
    consolidated_financials = pl.DataFrame(
        {
            "ticker": ["AAPL.US", "AAPL.US", "AAPL.US", "AAPL.US", "AAPL.US"],
            "statement": ["income_statement", "income_statement", "balance_sheet", "cash_flow", "shares"],
            "metric": ["revenue", "net_income", "total_assets", "free_cash_flow", "outstanding_shares"],
            "date": ["2025-03-31"] * 5,
            "filing_date": ["2025-05-01"] * 5,
            "value": [100.0, 20.0, 500.0, 50.0, 10_000_000.0],
            "source": ["open_source_consolidated"] * 5,
            "source_label": ["value"] * 5,
            "selected_source": ["sec_companyfacts"] * 5,
            "selected_source_label": ["tag"] * 5,
            "selected_form": ["10-Q"] * 5,
            "selected_fiscal_period": ["Q1"] * 5,
            "selected_fiscal_year": [2025] * 5,
            "source_priority": [1] * 5,
            "fallback_used": [False] * 5,
            "candidate_source_count": [1] * 5,
            "candidate_sources": ["sec_companyfacts"] * 5,
            "candidate_source_labels": ["tag"] * 5,
        }
    )
    earnings = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "reportDate": ["2025-05-01"],
            "earningsDatetime": ["2025-05-01 20:00:00"],
            "period_end": ["2025-03-31"],
            "epsEstimate": [1.4],
            "epsActual": [1.5],
            "surprisePercent": [7.0],
            "source": ["yfinance"],
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
    )

    income = pl.read_parquet(output_dir / "US_Income_statement.parquet")
    balance = pl.read_parquet(output_dir / "US_Balance_sheet.parquet")
    shares = pl.read_parquet(output_dir / "US_share.parquet")
    general = pl.read_parquet(output_dir / "US_General.parquet")
    earnings_export = pl.read_parquet(output_dir / "US_Earnings.parquet")

    assert "totalRevenue" in income.columns
    assert income["totalRevenue"].to_list() == [100.0]
    assert balance["commonStockSharesOutstanding"].to_list() == [10_000_000.0]
    assert shares["shares"].to_list() == [10_000_000.0]
    assert shares["sharesMln"].to_list() == [10.0]


def test_export_legacy_compatible_outputs_preserves_sec_period_end_for_earnings(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "live" / "legacy"
    reference_dir.mkdir(parents=True)
    _write_minimal_legacy_reference(reference_dir, ticker="TSCO.US", code="TSCO", name="Tractor Supply")

    clean_prices = pl.read_parquet(reference_dir / "US_Finalprice.parquet")
    benchmark_prices = pl.read_parquet(reference_dir / "SP500Price.parquet")
    general_reference = pl.DataFrame(
        {
            "ticker": ["TSCO.US"],
            "name": ["Tractor Supply"],
            "exchange": ["NASDAQ"],
            "cik": ["0000000000"],
            "source": ["sec_mapping"],
            "Sector": ["Consumer Defensive"],
            "industry": ["Specialty Retail"],
            "sector_source": ["sec_sic"],
            "sector_raw_value": [None],
            "sic": [None],
            "sic_description": [None],
            "mapping_rule": ["sec_sic:test"],
        }
    )
    consolidated_financials = pl.DataFrame(
        {
            "ticker": ["TSCO.US", "TSCO.US"],
            "statement": ["income_statement", "income_statement"],
            "metric": ["revenue", "net_income"],
            "date": ["2025-02-02", "2025-02-02"],
            "filing_date": ["2025-03-05", "2025-03-05"],
            "value": [100.0, 10.0],
            "source": ["open_source_consolidated", "open_source_consolidated"],
            "source_label": ["value", "value"],
            "selected_source": ["sec_companyfacts", "sec_companyfacts"],
            "selected_source_label": ["tag", "tag"],
            "selected_form": ["10-Q", "10-Q"],
            "selected_fiscal_period": ["Q1", "Q1"],
            "selected_fiscal_year": [2025, 2025],
            "source_priority": [1, 1],
            "fallback_used": [False, False],
            "candidate_source_count": [1, 1],
            "candidate_sources": ["sec_companyfacts", "sec_companyfacts"],
            "candidate_source_labels": ["tag", "tag"],
        }
    )
    earnings = pl.DataFrame(
        {
            "ticker": ["TSCO.US"],
            "reportDate": ["2025-03-05"],
            "earningsDatetime": ["2025-03-05 20:00:00"],
            "period_end": ["2025-02-02"],
            "epsEstimate": [1.4],
            "epsActual": [1.5],
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

    income = pl.read_parquet(output_dir / "US_Income_statement.parquet")
    earnings_export = pl.read_parquet(output_dir / "US_Earnings.parquet")

    assert income["date"].to_list() == ["2025-03-31"]
    assert earnings_export["date"].to_list() == ["2025-02-02"]


def test_export_legacy_compatible_outputs_aligns_balance_shares_with_earnings_semantics(tmp_path: Path) -> None:
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
            "source": ["open_source_general"],
            "Sector": ["Industrials"],
            "industry": ["Industrial Machinery"],
            "sector_source": ["yfinance"],
            "sector_raw_value": ["Industrials"],
            "sic": [None],
            "sic_description": [None],
            "mapping_rule": ["yfinance:sector"],
        }
    )
    consolidated_financials = pl.DataFrame(
        {
            "ticker": ["ABC.US", "ABC.US", "ABC.US", "ABC.US"],
            "statement": ["income_statement", "balance_sheet", "cash_flow", "shares"],
            "metric": ["net_income", "total_assets", "free_cash_flow", "outstanding_shares"],
            "date": ["2025-03-31"] * 4,
            "filing_date": ["2025-05-01"] * 4,
            "value": [20_000_000.0, 500_000_000.0, 50_000_000.0, 9_500_000.0],
            "source": ["open_source_consolidated"] * 4,
            "source_label": ["value"] * 4,
            "selected_source": ["sec_companyfacts", "sec_companyfacts", "sec_companyfacts", "sec_companyfacts"],
            "selected_source_label": ["tag"] * 4,
            "selected_form": ["10-Q"] * 4,
            "selected_fiscal_period": ["Q1"] * 4,
            "selected_fiscal_year": [2025] * 4,
            "source_priority": [1] * 4,
            "fallback_used": [False] * 4,
            "candidate_source_count": [1] * 4,
            "candidate_sources": ["sec_companyfacts"] * 4,
            "candidate_source_labels": ["tag"] * 4,
        }
    )
    earnings = pl.DataFrame(
        {
            "ticker": ["ABC.US"],
            "period_end": ["2025-03-31"],
            "reportDate": ["2025-05-01"],
            "earningsDatetime": ["2025-05-01 20:00:00"],
            "epsEstimate": [1.8],
            "epsActual": [2.0],
            "surprisePercent": [11.1],
            "selected_source": ["sec_submissions+yfinance"],
            "candidate_sources": ["sec_submissions | yfinance"],
            "calendar_source": ["sec_submissions"],
            "actual_source": ["yfinance"],
            "estimate_source": ["yfinance"],
            "surprise_source": ["yfinance"],
            "source_label": ["sec calendar + yahoo earnings"],
            "accession_number": ["0000000001-25-000001"],
            "form": ["10-Q"],
            "fiscal_period": ["Q1"],
            "fiscal_year": [2025],
        }
    )

    export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=benchmark_prices,
        general_reference=general_reference,
        consolidated_financials=consolidated_financials,
        consolidated_lineage=consolidated_financials,
        earnings_frame=earnings,
        reference_data_dir=reference_dir,
        output_dir=output_dir,
    )

    balance = pl.read_parquet(output_dir / "US_Balance_sheet.parquet")
    share_lineage = pl.read_parquet(output_dir / "lineage" / "legacy_share_semantics.parquet")

    assert balance["commonStockSharesOutstanding"].to_list() == [10_000_000.0]
    assert share_lineage["selected_method"].to_list() == ["earnings_implied"]
    assert share_lineage["reported_commonStockSharesOutstanding"].to_list() == [9_500_000.0]
    assert share_lineage["earnings_implied_commonStockSharesOutstanding"].to_list() == [10_000_000.0]
    assert share_lineage["actual_source"].to_list() == ["yfinance"]


def test_export_legacy_compatible_outputs_preserves_exact_earnings_period_end(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "live" / "legacy"
    reference_dir.mkdir(parents=True)

    pl.DataFrame(
        {
            "ticker": ["WDC.US"],
            "date": ["2025-01-01"],
            "adjusted_close": [1.0],
            "close": [1.0],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "volume": [10.0],
        }
    ).write_parquet(reference_dir / "US_Finalprice.parquet")
    pl.DataFrame(
        {
            "ticker": ["SPY.US"],
            "date": ["2025-01-01"],
            "adjusted_close": [1.0],
            "close": [1.0],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "volume": [10.0],
        }
    ).write_parquet(reference_dir / "SP500Price.parquet")
    pl.DataFrame({"Code": ["WDC"], "Name": ["Western Digital"], "Exchange": ["NASDAQ"], "CurrencyCode": ["USD"], "CurrencySymbol": ["$"], "CIK": ["0000106040"], "Sector": ["Technology"], "Industry": ["Storage"]}).write_parquet(
        reference_dir / "US_General.parquet"
    )
    pl.DataFrame({"ticker": ["WDC.US"], "date": ["2025-12-31"], "filing_date": ["2026-01-29"], "totalRevenue": [100.0], "netIncome": [20.0]}).write_parquet(
        reference_dir / "US_Income_statement.parquet"
    )
    pl.DataFrame(
        {
            "ticker": ["WDC.US"],
            "date": ["2025-12-31"],
            "filing_date": ["2026-01-29"],
            "commonStockSharesOutstanding": ["0.0"],
            "totalAssets": [500.0],
            "totalLiab": [300.0],
        }
    ).write_parquet(reference_dir / "US_Balance_sheet.parquet")
    pl.DataFrame({"ticker": ["WDC.US"], "date": ["2025-12-31"], "filing_date": ["2026-01-29"], "freeCashFlow": [50.0]}).write_parquet(
        reference_dir / "US_Cash_flow.parquet"
    )
    pl.DataFrame({"ticker": ["WDC.US"], "date": ["2025-12-31"], "dateFormatted": ["2025-12-31"], "sharesMln": [10.0], "shares": [10_000_000.0]}).write_parquet(
        reference_dir / "US_share.parquet"
    )
    pl.DataFrame(
        {
            "ticker": ["WDC.US"],
            "beforeAfterMarket": ["AfterMarket"],
            "currency": ["USD"],
            "date": ["2025-12-31"],
            "epsActual": [2.13],
            "epsDifference": [0.20],
            "epsEstimate": [1.93],
            "reportDate": ["2026-01-29"],
            "surprisePercent": [10.36],
        }
    ).write_parquet(reference_dir / "US_Earnings.parquet")

    clean_prices = pl.read_parquet(reference_dir / "US_Finalprice.parquet")
    benchmark_prices = pl.read_parquet(reference_dir / "SP500Price.parquet")
    general_reference = pl.DataFrame(
        {
            "ticker": ["WDC.US"],
            "name": ["Western Digital"],
            "exchange": ["NASDAQ"],
            "cik": ["0000106040"],
            "source": ["open_source_general"],
            "Sector": ["Technology"],
            "industry": ["Storage"],
            "sector_source": ["yfinance"],
            "sector_raw_value": ["Technology"],
            "sic": [None],
            "sic_description": [None],
            "mapping_rule": ["yfinance:sector"],
        }
    )
    consolidated_financials = pl.DataFrame(
        {
            "ticker": ["WDC.US", "WDC.US", "WDC.US", "WDC.US", "WDC.US"],
            "statement": ["income_statement", "income_statement", "balance_sheet", "cash_flow", "shares"],
            "metric": ["revenue", "net_income", "total_assets", "free_cash_flow", "outstanding_shares"],
            "date": ["2025-12-31"] * 5,
            "filing_date": ["2026-01-29"] * 5,
            "value": [100.0, 20.0, 500.0, 50.0, 10_000_000.0],
            "source": ["open_source_consolidated"] * 5,
            "source_label": ["value"] * 5,
            "selected_source": ["sec_companyfacts"] * 5,
            "selected_source_label": ["tag"] * 5,
            "selected_form": ["10-Q"] * 5,
            "selected_fiscal_period": ["Q2"] * 5,
            "selected_fiscal_year": [2026] * 5,
            "source_priority": [1] * 5,
            "fallback_used": [False] * 5,
            "candidate_source_count": [1] * 5,
            "candidate_sources": ["sec_companyfacts"] * 5,
            "candidate_source_labels": ["tag"] * 5,
        }
    )
    earnings = pl.DataFrame(
        {
            "ticker": ["WDC.US"],
            "reportDate": ["2026-01-29"],
            "earningsDatetime": ["2026-01-29 21:00:00"],
            "period_end": ["2026-01-02"],
            "epsEstimate": [1.93],
            "epsActual": [2.13],
            "surprisePercent": [10.36],
            "selected_source": ["sec_submissions+yfinance"],
            "candidate_sources": ["sec_submissions | yfinance"],
            "calendar_source": ["sec_submissions"],
            "actual_source": ["yfinance"],
            "estimate_source": ["yfinance"],
            "surprise_source": ["yfinance"],
            "source_label": ["calendar=sec_submissions | actual=yfinance | estimate=yfinance"],
            "accession_number": ["0001"],
            "form": ["10-Q"],
            "fiscal_period": ["Q2"],
            "fiscal_year": [2026],
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
    )

    earnings_export = pl.read_parquet(output_dir / "US_Earnings.parquet")

    assert earnings_export["date"].to_list() == ["2026-01-02"]


def test_export_legacy_compatible_outputs_normalizes_statement_dates_to_calendar_quarter_end(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    output_dir = tmp_path / "live" / "legacy"
    reference_dir.mkdir(parents=True)
    _write_minimal_legacy_reference(reference_dir, ticker="WDC.US", code="WDC", name="Western Digital")

    clean_prices = pl.read_parquet(reference_dir / "US_Finalprice.parquet")
    benchmark_prices = pl.read_parquet(reference_dir / "SP500Price.parquet")
    general_reference = pl.DataFrame(
        {
            "ticker": ["WDC.US"],
            "name": ["Western Digital"],
            "exchange": ["NASDAQ"],
            "cik": ["0000106040"],
            "source": ["open_source_general"],
            "Sector": ["Technology"],
            "industry": ["Storage"],
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
            "ticker": ["WDC.US", "WDC.US"],
            "statement": ["income_statement", "income_statement"],
            "metric": ["revenue", "revenue"],
            "date": ["2025-10-03", "2025-09-30"],
            "filing_date": ["2025-10-31", None],
            "value": [2_818_000_000.0, 2_818_000_000.0],
            "source": ["sec_companyfacts", "yfinance"],
            "source_label": ["tag", "statement row"],
            "selected_source": ["sec_companyfacts", "yfinance"],
            "selected_source_label": ["tag", "statement row"],
            "selected_form": ["10-Q", None],
            "selected_fiscal_period": ["Q1", None],
            "selected_fiscal_year": [2026, None],
            "source_priority": [1, 4],
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

    income = pl.read_parquet(output_dir / "US_Income_statement.parquet")
    assert income["date"].to_list() == ["2025-12-31"]
    assert income["filing_date"].to_list() == ["2025-10-31"]
    assert income["totalRevenue"].to_list() == [2_818_000_000.0]
