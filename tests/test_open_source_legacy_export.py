from __future__ import annotations

from pathlib import Path

import polars as pl

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
    pl.DataFrame({"Code": ["AAPL"], "Name": ["Apple"], "Exchange": ["NASDAQ"], "CurrencyCode": ["USD"], "CurrencySymbol": ["$"], "CIK": ["0000320193"]}).write_parquet(
        reference_dir / "US_General.parquet"
    )
    pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-03-31"], "filing_date": ["2025-05-01"], "totalRevenue": [100.0], "netIncome": [20.0]}).write_parquet(
        reference_dir / "US_Income_statement.parquet"
    )
    pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-03-31"], "filing_date": ["2025-05-01"], "totalAssets": [500.0], "totalLiab": [300.0]}).write_parquet(
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
            "beforeAfterMarket": [None],
            "currency": [None],
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
        {"ticker": ["AAPL.US"], "name": ["Apple"], "exchange": ["NASDAQ"], "cik": ["0000320193"], "source": ["sec_mapping"]}
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
    shares = pl.read_parquet(output_dir / "US_share.parquet")
    earnings_export = pl.read_parquet(output_dir / "US_Earnings.parquet")

    assert "totalRevenue" in income.columns
    assert income["totalRevenue"].to_list() == [100.0]
    assert shares["shares"].to_list() == [10_000_000.0]
    assert shares["sharesMln"].to_list() == [10.0]
    assert earnings_export["epsDifference"].to_list() == [0.10000000000000009]
