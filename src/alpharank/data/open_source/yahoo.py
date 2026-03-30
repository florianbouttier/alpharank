from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Iterable, Sequence

import pandas as pd
import polars as pl
import yfinance as yf

from alpharank.data.open_source.config import GENERAL_COLUMNS, PRICE_COLUMNS, specs_for_statement


class YahooFinanceClient:
    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        self._ticker_cache: dict[str, yf.Ticker] = {}
        self._cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(self._cache_dir))

    def _ticker(self, symbol: str) -> yf.Ticker:
        ticker = self._ticker_cache.get(symbol)
        if ticker is None:
            ticker = yf.Ticker(symbol)
            self._ticker_cache[symbol] = ticker
        return ticker

    def download_prices(self, tickers: Iterable[str], start_date: str, end_date: str, chunk_size: int = 5) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        tickers = list(tickers)
        for start_idx in range(0, len(tickers), chunk_size):
            chunk = tickers[start_idx : start_idx + chunk_size]
            history = _download_with_retries(chunk, start_date, end_date)
            for ticker in chunk:
                frame = _extract_price_frame(history, ticker)
                if frame is None:
                    frame = _download_single_ticker_frame(ticker, start_date, end_date)
                if frame is not None:
                    frames.append(frame)

        return pl.concat(frames, how="vertical") if frames else pl.DataFrame(schema={c: pl.String for c in PRICE_COLUMNS})

    def fetch_company_metadata(self, tickers: Iterable[str], max_workers: int = 2) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        ticker_list = list(tickers)
        if not ticker_list:
            return pl.DataFrame(
                schema={
                    "ticker": pl.String,
                    "name": pl.String,
                    "exchange": pl.String,
                    "sector_raw_value": pl.String,
                    "industry": pl.String,
                }
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._fetch_ticker_metadata, ticker): ticker for ticker in ticker_list}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception:
                    row = None
                if row is not None:
                    rows.append(row)

        if not rows:
            return pl.DataFrame(
                schema={
                    "ticker": pl.String,
                    "name": pl.String,
                    "exchange": pl.String,
                    "sector_raw_value": pl.String,
                    "industry": pl.String,
                }
            )
        return pl.DataFrame(rows).sort("ticker")

    def fetch_general_reference(self, tickers: Iterable[str], sec_mapping: pl.DataFrame) -> pl.DataFrame:
        mapping = sec_mapping.with_columns(pl.col("cik").cast(pl.Utf8).str.zfill(10))
        yahoo_metadata = self.fetch_company_metadata(tickers)
        rows: list[dict[str, object]] = []
        for ticker in tickers:
            match = mapping.filter(pl.col("ticker") == ticker).select(["name", "exchange", "cik"]).to_dicts()
            if not match:
                continue
            item = match[0]
            yahoo_match = (
                yahoo_metadata.filter(pl.col("ticker") == f"{ticker}.US")
                .select(["name", "exchange", "sector_raw_value", "industry"])
                .to_dicts()
            )
            yahoo_item = yahoo_match[0] if yahoo_match else {}
            rows.append(
                {
                    "ticker": f"{ticker}.US",
                    "name": str(yahoo_item.get("name") or item["name"]),
                    "exchange": str(item["exchange"]),
                    "cik": str(item["cik"]),
                    "source": "open_source_general",
                    "Sector": str(yahoo_item.get("sector_raw_value")) if yahoo_item.get("sector_raw_value") is not None else None,
                    "industry": str(yahoo_item.get("industry")) if yahoo_item.get("industry") is not None else None,
                    "sector_source": "yfinance" if yahoo_item.get("sector_raw_value") is not None else None,
                    "sector_raw_value": str(yahoo_item.get("sector_raw_value")) if yahoo_item.get("sector_raw_value") is not None else None,
                    "sic": None,
                    "sic_description": None,
                    "mapping_rule": "yfinance:sector" if yahoo_item.get("sector_raw_value") is not None else None,
                }
            )
        return pl.DataFrame(rows).select(GENERAL_COLUMNS) if rows else pl.DataFrame(schema={c: pl.String for c in GENERAL_COLUMNS})

    def fetch_earnings_dates(self, tickers: Iterable[str], limit: int = 8) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for ticker in tickers:
            history = _safe_get_earnings_dates(self._ticker(ticker), ticker=ticker, limit=limit)
            if history is None or history.empty:
                continue
            frame = history.reset_index()
            date_col = frame.columns[0]
            if date_col not in frame.columns:
                continue
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

    def fetch_quarterly_financials(self, tickers: Iterable[str], max_workers: int = 2) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        ticker_list = list(tickers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_ticker_financial_frames, ticker): ticker for ticker in ticker_list}
            for future in as_completed(futures):
                try:
                    ticker_frames = future.result()
                except Exception:
                    ticker_frames = []
                if ticker_frames:
                    frames.extend(ticker_frames)
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
                        pl.coalesce([pl.col("period_end"), pl.col("reportDate")]).alias("date"),
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
        chunk_size: int = 20,
    ) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for start_idx in range(0, len(tickers), chunk_size):
            chunk = list(tickers[start_idx : start_idx + chunk_size])
            history = _download_with_retries(chunk, start_date, end_date)
            for ticker in chunk:
                available = _history_has_prices(history, ticker)
                if not available:
                    single = _download_with_retries([ticker], start_date, end_date)
                    available = _history_has_prices(single, ticker)
                rows.append(
                    {
                        "ticker": f"{ticker}.US",
                        "ticker_root": ticker,
                        "yahoo_price_available": available,
                    }
                )
        return pl.DataFrame(rows).sort("ticker") if rows else pl.DataFrame(
            schema={"ticker": pl.String, "ticker_root": pl.String, "yahoo_price_available": pl.Boolean}
        )

    def _fetch_ticker_metadata(self, ticker: str) -> dict[str, object] | None:
        info = _safe_get_info(self._ticker(ticker), ticker=ticker)
        if not info:
            return None
        sector = info.get("sectorDisp") or info.get("sector")
        industry = info.get("industryDisp") or info.get("industry")
        exchange = info.get("fullExchangeName") or info.get("exchange")
        name = info.get("longName") or info.get("shortName") or info.get("displayName")
        if not any(value is not None for value in (name, exchange, sector, industry)):
            return None
        return {
            "ticker": f"{ticker}.US",
            "name": str(name) if name is not None else None,
            "exchange": str(exchange) if exchange is not None else None,
            "sector_raw_value": str(sector) if sector is not None else None,
            "industry": str(industry) if industry is not None else None,
        }


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


def _fetch_ticker_financial_frames(ticker: str) -> list[pl.DataFrame]:
    ticker_obj = yf.Ticker(ticker)
    statement_map = {
        "income_statement": lambda current: current.quarterly_income_stmt,
        "balance_sheet": lambda current: current.quarterly_balance_sheet,
        "cash_flow": lambda current: current.quarterly_cashflow,
        "shares": lambda current: current.quarterly_balance_sheet,
    }
    frames: list[pl.DataFrame] = []
    for statement, getter in statement_map.items():
        try:
            wide = getter(ticker_obj)
        except Exception:
            continue
        if wide is None or wide.empty:
            continue
        frame = _extract_statement_frame(ticker, statement, wide)
        if not frame.is_empty():
            frames.append(frame)
    return frames


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
        try:
            history = yf.download(
                chunk,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="ticker",
            )
        except Exception:
            history = pd.DataFrame()
        last_history = history
        if not history.empty:
            return history
        time.sleep(2 * (attempt + 1))
    return last_history


def _download_single_ticker_frame(ticker: str, start_date: str, end_date: str) -> pl.DataFrame | None:
    history = _download_with_retries([ticker], start_date, end_date)
    return _extract_price_frame(history, ticker)


def _safe_get_earnings_dates(ticker_obj: yf.Ticker, *, ticker: str, limit: int) -> pd.DataFrame | None:
    try:
        history = ticker_obj.get_earnings_dates(limit=limit)
    except Exception as exc:
        print(f"{ticker}: earnings fetch skipped ({exc})")
        return None
    if history is None or history.empty:
        return history
    if "Earnings Date" not in history.columns and getattr(history.index, "name", None) != "Earnings Date":
        return None
    return history


def _safe_get_info(ticker_obj: yf.Ticker, *, ticker: str) -> dict[str, object] | None:
    try:
        info = ticker_obj.get_info()
    except Exception as exc:
        print(f"{ticker}: metadata fetch skipped ({exc})")
        return None
    return info if isinstance(info, dict) else None


def _history_has_prices(history: pd.DataFrame, ticker: str) -> bool:
    if history is None or history.empty:
        return False
    if isinstance(history.columns, pd.MultiIndex):
        if ticker not in history.columns.get_level_values(0):
            return False
        ticker_frame = history[ticker]
        return not ticker_frame.dropna(how="all").empty
    return not history.dropna(how="all").empty
