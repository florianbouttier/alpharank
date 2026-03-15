#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def _norm_ticker_from_constituents(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: re.sub(r"\\.", "-", x) if isinstance(x, str) else x).astype(str) + ".US"


def main(
    *,
    source_data_dir: str | Path | None = None,
    target_data_dir: str | Path | None = None,
    start_date: str = "2019-01-01",
    end_date: str = "2022-12-31",
    max_tickers: int = 80,
    fundamentals_lookback_days: int = 730,
) -> None:
    benchmark_repo_root = Path(__file__).resolve().parents[1]
    src = (
        Path(source_data_dir).expanduser().resolve()
        if source_data_dir
        else benchmark_repo_root.parents[1] / "data"
    )
    dst = (
        Path(target_data_dir).expanduser().resolve()
        if target_data_dir
        else benchmark_repo_root / "data" / "subset_legacy_v1"
    )
    dst.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    fundamentals_start = start - pd.Timedelta(days=fundamentals_lookback_days)

    final_price = pd.read_parquet(src / "US_Finalprice.parquet")
    final_price["date"] = pd.to_datetime(final_price["date"], errors="coerce")
    fp_window = final_price[(final_price["date"] >= start) & (final_price["date"] <= end)].copy()

    top = (
        fp_window.groupby("ticker", as_index=False)
        .size()
        .sort_values(["size", "ticker"], ascending=[False, True])
        .head(max_tickers)
    )
    tickers = sorted(top["ticker"].tolist())

    # price files
    final_price_sub = final_price[(final_price["ticker"].isin(tickers)) & (final_price["date"] >= start) & (final_price["date"] <= end)].copy()
    sp500_price = pd.read_parquet(src / "SP500Price.parquet")
    sp500_price["date"] = pd.to_datetime(sp500_price["date"], errors="coerce")
    sp500_price_sub = sp500_price[(sp500_price["date"] >= start) & (sp500_price["date"] <= end)].copy()

    # general
    general = pd.read_parquet(src / "US_General.parquet")
    general_sub = general[general["ticker"].isin(tickers)].copy()

    # fundamentals
    balance = pd.read_parquet(src / "US_Balance_sheet.parquet")
    cash = pd.read_parquet(src / "US_Cash_flow.parquet")
    income = pd.read_parquet(src / "US_Income_statement.parquet")
    earnings = pd.read_parquet(src / "US_Earnings.parquet")

    for df in (balance, cash, income, earnings):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    balance_sub = balance[(balance["ticker"].isin(tickers)) & (balance["date"] >= fundamentals_start) & (balance["date"] <= end)].copy()
    cash_sub = cash[(cash["ticker"].isin(tickers)) & (cash["date"] >= fundamentals_start) & (cash["date"] <= end)].copy()
    income_sub = income[(income["ticker"].isin(tickers)) & (income["date"] >= fundamentals_start) & (income["date"] <= end)].copy()
    earnings_sub = earnings[(earnings["ticker"].isin(tickers)) & (earnings["date"] >= fundamentals_start) & (earnings["date"] <= end)].copy()

    # constituents (keep original schema expected by run_legacy)
    constituents = pd.read_csv(src / "SP500_Constituents.csv")
    constituents["Date"] = pd.to_datetime(constituents["Date"], errors="coerce")
    constituents["ticker_norm"] = _norm_ticker_from_constituents(constituents["Ticker"])
    constituents_sub = constituents[
        (constituents["Date"] >= fundamentals_start)
        & (constituents["Date"] <= end)
        & (constituents["ticker_norm"].isin(tickers))
    ].drop(columns=["ticker_norm"])

    # write subset files with expected names
    final_price_sub.to_parquet(dst / "US_Finalprice.parquet", index=False)
    general_sub.to_parquet(dst / "US_General.parquet", index=False)
    income_sub.to_parquet(dst / "US_Income_statement.parquet", index=False)
    balance_sub.to_parquet(dst / "US_Balance_sheet.parquet", index=False)
    cash_sub.to_parquet(dst / "US_Cash_flow.parquet", index=False)
    earnings_sub.to_parquet(dst / "US_Earnings.parquet", index=False)
    constituents_sub.to_csv(dst / "SP500_Constituents.csv", index=False)
    sp500_price_sub.to_parquet(dst / "SP500Price.parquet", index=False)

    metadata = {
        "source_data_dir": str(src),
        "target_data_dir": str(dst),
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "fundamentals_start_date": str(fundamentals_start.date()),
        "max_tickers": max_tickers,
        "selected_tickers": tickers,
        "rows": {
            "US_Finalprice": int(len(final_price_sub)),
            "US_General": int(len(general_sub)),
            "US_Income_statement": int(len(income_sub)),
            "US_Balance_sheet": int(len(balance_sub)),
            "US_Cash_flow": int(len(cash_sub)),
            "US_Earnings": int(len(earnings_sub)),
            "SP500_Constituents": int(len(constituents_sub)),
            "SP500Price": int(len(sp500_price_sub)),
        },
    }
    (dst / "subset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
