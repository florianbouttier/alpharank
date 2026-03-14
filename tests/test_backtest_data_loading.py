from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.backtest.data_loading import load_raw_data


def test_load_raw_data_uses_price_overrides_without_touching_financials(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    default_final_price = pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-01-31"], "adjusted_close": [1.0]})
    override_final_price = pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-01-31"], "adjusted_close": [2.0]})
    default_sp500 = pl.DataFrame({"ticker": ["SPY.US"], "date": ["2025-01-31"], "close": [10.0]})
    override_sp500 = pl.DataFrame({"ticker": ["SPY.US"], "date": ["2025-01-31"], "close": [20.0]})
    income_statement = pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-03-31"], "netIncome": [100.0]})
    balance_sheet = pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-03-31"], "totalAssets": [200.0]})
    cash_flow = pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-03-31"], "freeCashFlow": [50.0]})
    earnings = pl.DataFrame({"ticker": ["AAPL.US"], "date": ["2025-03-31"], "reportDate": ["2025-05-01"]})
    constituents = "Ticker,Date\nAAPL,2025-01-31\n"

    default_final_price.write_parquet(data_dir / "US_Finalprice.parquet")
    default_sp500.write_parquet(data_dir / "SP500Price.parquet")
    income_statement.write_parquet(data_dir / "US_Income_statement.parquet")
    balance_sheet.write_parquet(data_dir / "US_Balance_sheet.parquet")
    cash_flow.write_parquet(data_dir / "US_Cash_flow.parquet")
    earnings.write_parquet(data_dir / "US_Earnings.parquet")
    (data_dir / "SP500_Constituents.csv").write_text(constituents, encoding="utf-8")

    override_dir = tmp_path / "override"
    override_dir.mkdir()
    override_final_price.write_parquet(override_dir / "US_Finalprice.parquet")
    override_sp500.write_parquet(override_dir / "SP500Price.parquet")

    raw = load_raw_data(
        data_dir,
        final_price_path=override_dir / "US_Finalprice.parquet",
        sp500_price_path=override_dir / "SP500Price.parquet",
    )

    assert raw.final_price["adjusted_close"].to_list() == [2.0]
    assert raw.sp500_price["close"].to_list() == [20.0]
    assert raw.income_statement["netIncome"].to_list() == [100.0]
    assert raw.balance_sheet["totalAssets"].to_list() == [200.0]
    assert raw.cash_flow["freeCashFlow"].to_list() == [50.0]
