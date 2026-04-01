from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.data.open_source.legacy_export import export_legacy_compatible_outputs


def _write_minimal_legacy_reference(reference_dir: Path, *, ticker: str, code: str, name: str) -> None:
    pl.DataFrame(
        {
            "ticker": [ticker],
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
    pl.DataFrame(
        {
            "Code": [code],
            "Name": [name],
            "Exchange": ["NASDAQ"],
            "CurrencyCode": ["USD"],
            "CurrencySymbol": ["$"],
            "CIK": ["0000000000"],
            "Sector": ["Technology"],
            "Industry": [""],
        }
    ).write_parquet(reference_dir / "US_General.parquet")
    pl.DataFrame({"ticker": [ticker], "date": ["2025-03-31"], "filing_date": ["2025-05-01"], "totalRevenue": [100.0]}).write_parquet(
        reference_dir / "US_Income_statement.parquet"
    )
    pl.DataFrame(
        {
            "ticker": [ticker],
            "date": ["2025-03-31"],
            "filing_date": ["2025-05-01"],
            "commonStockSharesOutstanding": ["0.0"],
            "totalAssets": [500.0],
        }
    ).write_parquet(reference_dir / "US_Balance_sheet.parquet")
    pl.DataFrame({"ticker": [ticker], "date": ["2025-03-31"], "filing_date": ["2025-05-01"], "freeCashFlow": [50.0]}).write_parquet(
        reference_dir / "US_Cash_flow.parquet"
    )
    pl.DataFrame(
        {"ticker": [ticker], "date": ["2025-03-31"], "dateFormatted": ["2025-03-31"], "sharesMln": [10.0], "shares": [10_000_000.0]}
    ).write_parquet(reference_dir / "US_share.parquet")
    pl.DataFrame(
        {
            "ticker": [ticker],
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
    assert general["Sector"].to_list() == ["Technology"]
    assert general["Industry"].to_list() == ["Consumer Electronics"]
    assert earnings_export["epsDifference"].to_list() == [0.10000000000000009]
    assert earnings_export["beforeAfterMarket"].to_list() == ["AfterMarket"]


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


def test_export_legacy_compatible_outputs_normalizes_earnings_period_end_to_legacy_month_end(tmp_path: Path) -> None:
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

    assert earnings_export["date"].to_list() == ["2025-12-31"]


def test_export_legacy_compatible_outputs_normalizes_statement_dates_to_month_end(tmp_path: Path) -> None:
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
    assert income["date"].to_list() == ["2025-09-30"]
    assert income["filing_date"].to_list() == ["2025-10-31"]
    assert income["totalRevenue"].to_list() == [2_818_000_000.0]


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
    assert income.filter(pl.col("date") == "2025-10-31")["totalRevenue"].to_list() == [1_419_000_000.0]
    assert income.filter(pl.col("date") == "2025-10-31")["filing_date"].to_list() == ["2025-12-17"]


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


def test_export_legacy_compatible_outputs_collapses_duplicate_dates_with_same_filing(tmp_path: Path) -> None:
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
    assert balance["date"].to_list() == ["2025-05-31"]
    assert balance["totalLiab"].to_list() == [56_902_764_000.0]
    assert balance["totalStockholderEquity"].to_list() == [10_305_025_000.0]


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
    assert cash["date"].to_list() == ["2025-02-28"]
    assert cash["filing_date"].to_list() == ["2025-03-20"]
    assert cash["freeCashFlow"].to_list() == [2_682_588_000.0]
