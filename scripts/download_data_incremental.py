from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent

from alpharank.data.lineage import create_snapshot
from alpharank.data.service import APIClient, EODHDDataService, PriceData


def _normalize_price_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out

    if "adjusted_close" not in out.columns and "close" in out.columns:
        out["adjusted_close"] = out["close"]
    out["ticker"] = ticker

    expected_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted_close",
        "ticker",
    ]
    for column in expected_columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out[expected_columns]


def _fetch_price_window(api_key: str, ticker: str, start_date: str, end_date: str) -> tuple[str, pd.DataFrame]:
    price = PriceData(APIClient(api_key))
    raw = price.get_raw_price_data(ticker, start_date=start_date, end_date=end_date)
    technical = price.get_technical_data(ticker, start_date=start_date, end_date=end_date)

    if raw.empty and technical.empty:
        return ticker, pd.DataFrame()
    if technical.empty:
        return ticker, _normalize_price_frame(raw, ticker)
    if raw.empty:
        return ticker, _normalize_price_frame(technical, ticker)

    merged = technical.merge(raw[["date", "adjusted_close"]], on="date", how="left")
    return ticker, _normalize_price_frame(merged, ticker)


def _merge_incremental(existing: pd.DataFrame, delta: pd.DataFrame, refresh_start: str) -> pd.DataFrame:
    if delta.empty:
        out = existing.copy()
    else:
        out = existing.copy()
        out["date"] = pd.to_datetime(out["date"])
        delta = delta.copy()
        delta["date"] = pd.to_datetime(delta["date"])
        delta_tickers = set(delta["ticker"].astype(str).unique())
        refresh_start_ts = pd.Timestamp(refresh_start)
        base = out[
            ~(
                out["ticker"].astype(str).isin(delta_tickers)
                & (out["date"] >= refresh_start_ts)
            )
        ]
        out = pd.concat([base, delta], ignore_index=True)

    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset=["ticker", "date"], keep="last")
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def _collect_price_delta(
    *,
    api_key: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
    max_workers: int,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_price_window, api_key, ticker, start_date, end_date): ticker
            for ticker in tickers
        }
        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                _, frame = future.result()
                if not frame.empty:
                    frames.append(frame)
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append(f"{ticker}: {exc}")

            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"Fetched price windows: {completed}/{total}")

    if frames:
        return pd.concat(frames, ignore_index=True), failures
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "adjusted_close", "ticker"]), failures


def main(
    *,
    refresh_days: int = 7,
    max_workers: int = 4,
    new_ticker_start_date: str = "2005-01-01",
) -> None:
    data_dir = project_root / "data"
    final_price_path = data_dir / "US_Finalprice.parquet"
    sp500_price_path = data_dir / "SP500Price.parquet"
    constituents_path = data_dir / "SP500_Constituents.csv"

    api_client = APIClient()
    api_key = api_client.get_api_key()
    if not api_key:
        raise RuntimeError("EODHD_API_KEY is missing")

    service = EODHDDataService(api_key)

    print("Refreshing S&P 500 historical composition...")
    historical_company_sp500 = service.get_sp500_historical_composition()
    print("Refreshing US exchange ticker list...")
    ticker_from_exchange = service.get_ticker_list_from_exchange(exchange_code="US")[["Code", "Type"]]

    historical_company = historical_company_sp500.merge(
        ticker_from_exchange,
        left_on=["Ticker"],
        right_on=["Code"],
        how="left",
    )
    historical_company = historical_company[
        (historical_company["Type"] == "Common Stock") | pd.isna(historical_company["Type"])
    ]
    historical_company = historical_company.dropna(subset=["Ticker"]).copy()
    historical_company["Ticker"] = [
        str(ticker).replace(".", "-") + ".US"
        for ticker in historical_company["Ticker"]
    ]
    tickers = sorted(historical_company["Ticker"].unique().tolist())
    print(f"Active tickers to refresh: {len(tickers)}")

    existing_prices = pd.read_parquet(final_price_path)
    existing_prices["date"] = pd.to_datetime(existing_prices["date"])
    last_price_date = existing_prices["date"].max().date()
    refresh_start_date = max(
        date(2005, 1, 1),
        last_price_date - timedelta(days=refresh_days),
    ).isoformat()
    end_date = date.today().isoformat()
    print(f"Incremental price window for existing tickers: {refresh_start_date} -> {end_date}")

    existing_tickers = set(existing_prices["ticker"].astype(str).unique())
    existing_active_tickers = [ticker for ticker in tickers if ticker in existing_tickers]
    new_tickers = [ticker for ticker in tickers if ticker not in existing_tickers]
    print(f"Existing tickers: {len(existing_active_tickers)} | New tickers: {len(new_tickers)}")

    delta_existing, failures_existing = _collect_price_delta(
        api_key=api_key,
        tickers=existing_active_tickers,
        start_date=refresh_start_date,
        end_date=end_date,
        max_workers=max_workers,
    )
    delta_new, failures_new = _collect_price_delta(
        api_key=api_key,
        tickers=new_tickers,
        start_date=new_ticker_start_date,
        end_date=end_date,
        max_workers=max_workers,
    )
    price_delta = pd.concat([delta_existing, delta_new], ignore_index=True)
    updated_prices = _merge_incremental(existing_prices, price_delta, refresh_start_date)

    existing_spy = pd.read_parquet(sp500_price_path)
    existing_spy["date"] = pd.to_datetime(existing_spy["date"])
    last_spy_date = existing_spy["date"].max().date()
    spy_refresh_start = max(
        date(2005, 1, 1),
        last_spy_date - timedelta(days=refresh_days),
    ).isoformat()
    print(f"Incremental SPY window: {spy_refresh_start} -> {end_date}")
    spy_delta, failures_spy = _collect_price_delta(
        api_key=api_key,
        tickers=["SPY"],
        start_date=spy_refresh_start,
        end_date=end_date,
        max_workers=1,
    )
    updated_spy = _merge_incremental(existing_spy, spy_delta, spy_refresh_start)

    print("Writing incremental outputs...")
    historical_company_sp500.to_csv(constituents_path, index=False)
    updated_prices.to_parquet(final_price_path, index=False)
    updated_spy.to_parquet(sp500_price_path, index=False)

    all_files = {
        "sp500_constituents": constituents_path,
        "us_finalprice": final_price_path,
        "us_general": data_dir / "US_General.parquet",
        "us_income_statement": data_dir / "US_Income_statement.parquet",
        "us_balance_sheet": data_dir / "US_Balance_sheet.parquet",
        "us_cash_flow": data_dir / "US_Cash_flow.parquet",
        "us_earnings": data_dir / "US_Earnings.parquet",
        "us_share": data_dir / "US_share.parquet",
        "sp500price": sp500_price_path,
    }
    manifest = create_snapshot(
        data_dir=data_dir,
        files=all_files,
        frames={
            "sp500_constituents": historical_company_sp500,
            "us_finalprice": updated_prices,
            "sp500price": updated_spy,
        },
    )

    failures = failures_existing + failures_new + failures_spy
    print(f"Updated equity price rows: {len(price_delta)}")
    print(f"Updated SPY price rows: {len(spy_delta)}")
    print(f"Snapshot created: {manifest['snapshot_dir']}")
    print(f"Latest manifest: {data_dir / 'latest_snapshot.json'}")
    if failures:
        print(f"Warnings: {len(failures)} price fetch failures retained previous history.")
        for failure in failures[:20]:
            print(f"  - {failure}")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} additional warnings omitted")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental EODHD price refresh for legacy datasets.")
    parser.add_argument("--refresh-days", type=int, default=7, help="Number of trailing calendar days to refetch for existing tickers.")
    parser.add_argument("--max-workers", type=int, default=4, help="Max concurrent ticker fetches for daily price refresh.")
    parser.add_argument(
        "--new-ticker-start-date",
        default="2005-01-01",
        help="Full-history start date for tickers absent from the existing price parquet.",
    )
    args = parser.parse_args()
    main(
        refresh_days=args.refresh_days,
        max_workers=args.max_workers,
        new_ticker_start_date=args.new_ticker_start_date,
    )
