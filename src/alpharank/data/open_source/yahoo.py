from __future__ import annotations

import time
from typing import Iterable, Sequence

import pandas as pd
import polars as pl
import yfinance as yf

from alpharank.data.open_source.config import GENERAL_COLUMNS, PRICE_COLUMNS, specs_for_statement


class YahooFinanceClient:
    def __init__(self) -> None:
        self._ticker_cache: dict[str, yf.Ticker] = {}

    def _ticker(self, symbol: str) -> yf.Ticker:
        ticker = self._ticker_cache.get(symbol)
        if ticker is None:
            ticker = yf.Ticker(symbol)
            self._ticker_cache[symbol] = ticker
        return ticker

    def download_prices(self, tickers: Iterable[str], start_date: str, end_date: str) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        tickers = list(tickers)
        for start_idx in range(0, len(tickers), 50):
            chunk = tickers[start_idx : start_idx + 50]
            history = _download_with_retries(chunk, start_date, end_date)
            for ticker in chunk:
                frame = _extract_price_frame(history, ticker)
                if frame is not None:
                    frames.append(frame)

        return pl.concat(frames, how="vertical") if frames else pl.DataFrame(schema={c: pl.String for c in PRICE_COLUMNS})

    def fetch_general_reference(self, tickers: Iterable[str], sec_mapping: pl.DataFrame) -> pl.DataFrame:
        mapping = sec_mapping.with_columns(pl.col("cik").cast(pl.Utf8).str.zfill(10))
        rows: list[dict[str, str]] = []
        for ticker in tickers:
            match = mapping.filter(pl.col("ticker") == ticker).select(["name", "exchange", "cik"]).to_dicts()
            if not match:
                continue
            item = match[0]
            rows.append(
                {
                    "ticker": f"{ticker}.US",
                    "name": str(item["name"]),
                    "exchange": str(item["exchange"]),
                    "cik": str(item["cik"]),
                    "source": "sec_mapping",
                }
            )
        return pl.DataFrame(rows).select(GENERAL_COLUMNS) if rows else pl.DataFrame(schema={c: pl.String for c in GENERAL_COLUMNS})

    def fetch_earnings_dates(self, tickers: Iterable[str], limit: int = 8) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for ticker in tickers:
            history = self._ticker(ticker).get_earnings_dates(limit=limit)
            if history is None or history.empty:
                continue
            frame = history.reset_index()
            date_col = frame.columns[0]
            for record in frame.to_dict(orient="records"):
                earnings_date = pd.Timestamp(record[date_col]).tz_localize(None) if getattr(record[date_col], "tzinfo", None) else pd.Timestamp(record[date_col])
                rows.append(
                    {
                        "ticker": f"{ticker}.US",
                        "reportDate": earnings_date.strftime("%Y-%m-%d"),
                        "earningsDatetime": earnings_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "period_end": None,
                        "epsEstimate": _as_float(record.get("EPS Estimate")),
                        "epsActual": _as_float(record.get("Reported EPS")),
                        "surprisePercent": _as_float(record.get("Surprise(%)")),
                        "source": "yfinance",
                    }
                )
        if not rows:
            return pl.DataFrame(
                schema={
                    "ticker": pl.String,
                    "reportDate": pl.String,
                    "earningsDatetime": pl.String,
                    "period_end": pl.String,
                    "epsEstimate": pl.Float64,
                    "epsActual": pl.Float64,
                    "surprisePercent": pl.Float64,
                    "source": pl.String,
                }
            )
        return pl.DataFrame(rows).sort(["ticker", "reportDate"])

    def fetch_quarterly_financials(self, tickers: Iterable[str]) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        statement_map = {
            "income_statement": lambda ticker_obj: ticker_obj.quarterly_income_stmt,
            "balance_sheet": lambda ticker_obj: ticker_obj.quarterly_balance_sheet,
            "cash_flow": lambda ticker_obj: ticker_obj.quarterly_cashflow,
            "shares": lambda ticker_obj: ticker_obj.quarterly_balance_sheet,
        }

        for ticker in tickers:
            ticker_obj = self._ticker(ticker)
            for statement, getter in statement_map.items():
                wide = getter(ticker_obj)
                if wide is None or wide.empty:
                    continue
                frames.append(_extract_statement_frame(ticker, statement, wide))

        return pl.concat(frames, how="vertical") if frames else _empty_financial_frame()

    def normalize_earnings_long(self, earnings_dates: pl.DataFrame) -> pl.DataFrame:
        if earnings_dates.is_empty():
            return _empty_financial_frame()

        frames: list[pl.DataFrame] = []
        metric_map = {
            "eps_actual": "epsActual",
            "eps_estimate": "epsEstimate",
            "surprise_percent": "surprisePercent",
        }
        for metric, column in metric_map.items():
            frames.append(
                earnings_dates.select(
                    [
                        pl.col("ticker"),
                        pl.lit("earnings").alias("statement"),
                        pl.lit(metric).alias("metric"),
                        pl.col("reportDate").alias("date"),
                        pl.col("reportDate").alias("filing_date"),
                        pl.col(column).cast(pl.Float64, strict=False).alias("value"),
                        pl.lit("yfinance").alias("source"),
                        pl.lit(column).alias("source_label"),
                    ]
                ).filter(pl.col("value").is_not_null())
            )
        return pl.concat(frames, how="vertical").sort(["ticker", "metric", "date"])

    def audit_price_availability(
        self,
        tickers: Sequence[str],
        start_date: str,
        end_date: str,
        chunk_size: int = 50,
    ) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for start_idx in range(0, len(tickers), chunk_size):
            chunk = list(tickers[start_idx : start_idx + chunk_size])
            history = yf.download(
                chunk,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
            )
            for ticker in chunk:
                rows.append(
                    {
                        "ticker": f"{ticker}.US",
                        "ticker_root": ticker,
                        "yahoo_price_available": _history_has_prices(history, ticker),
                    }
                )
        return pl.DataFrame(rows).sort("ticker") if rows else pl.DataFrame(
            schema={"ticker": pl.String, "ticker_root": pl.String, "yahoo_price_available": pl.Boolean}
        )


def _extract_statement_frame(ticker: str, statement: str, wide: pd.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs_for_statement(statement):
        label = next((candidate for candidate in spec.yfinance_rows if candidate in wide.index), None)
        if label is None:
            continue
        for column, value in wide.loc[label].items():
            if pd.isna(value):
                continue
            numeric_value = float(value)
            if spec.metric == "capital_expenditures":
                numeric_value = abs(numeric_value)
            rows.append(
                {
                    "ticker": f"{ticker}.US",
                    "statement": statement,
                    "metric": spec.metric,
                    "date": pd.Timestamp(column).strftime("%Y-%m-%d"),
                    "filing_date": None,
                    "value": numeric_value,
                    "source": "yfinance",
                    "source_label": label,
                }
            )
    return pl.DataFrame(rows) if rows else _empty_financial_frame()


def _empty_financial_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "filing_date": pl.String,
            "value": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
        }
    )


def _as_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _extract_price_frame(history: pd.DataFrame, ticker: str) -> pl.DataFrame | None:
    if history is None or history.empty:
        return None
    if isinstance(history.columns, pd.MultiIndex):
        if ticker not in history.columns.get_level_values(0):
            return None
        ticker_history = history[ticker]
    else:
        ticker_history = history
    if ticker_history.empty or ticker_history.dropna(how="all").empty:
        return None

    ticker_history = ticker_history.reset_index()
    ticker_history.columns = [col[0] if isinstance(col, tuple) else col for col in ticker_history.columns]
    frame = pl.from_pandas(ticker_history, include_index=False).rename(
        {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "Volume": "volume",
        }
    )
    return (
        frame.select(
            [
                pl.col("date").cast(pl.Date, strict=False).dt.strftime("%Y-%m-%d").alias("date"),
                pl.col("open").cast(pl.Float64, strict=False).alias("open"),
                pl.col("high").cast(pl.Float64, strict=False).alias("high"),
                pl.col("low").cast(pl.Float64, strict=False).alias("low"),
                pl.col("close").cast(pl.Float64, strict=False).alias("close"),
                pl.col("volume").cast(pl.Float64, strict=False).alias("volume"),
                pl.col("adjusted_close").cast(pl.Float64, strict=False).alias("adjusted_close"),
            ]
        )
        .with_columns(pl.lit(f"{ticker}.US").alias("ticker"))
        .select(PRICE_COLUMNS)
    )


def _download_with_retries(chunk: list[str], start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    last_history = pd.DataFrame()
    for attempt in range(retries):
        history = yf.download(
            chunk,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
        last_history = history
        if not history.empty:
            return history
        time.sleep(2 * (attempt + 1))
    return last_history


def _history_has_prices(history: pd.DataFrame, ticker: str) -> bool:
    if history is None or history.empty:
        return False
    if isinstance(history.columns, pd.MultiIndex):
        if ticker not in history.columns.get_level_values(0):
            return False
        ticker_frame = history[ticker]
        return not ticker_frame.dropna(how="all").empty
    return not history.dropna(how="all").empty
