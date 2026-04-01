from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
import json
import os

from dotenv import load_dotenv
import polars as pl
import requests


API_BASE_URL = "https://stockanalysis.com/api"


class StockAnalysisClient:
    """Lightweight client for StockAnalysis historical price API."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        max_workers: int = 4,
        timeout_seconds: int = 30,
        user_agent: str | None = None,
    ) -> None:
        _load_local_dotenv()
        self.cache_dir = cache_dir or (Path.cwd() / "data" / "open_source" / "_cache" / "stockanalysis")
        self.max_workers = max(1, max_workers)
        self.timeout_seconds = timeout_seconds
        self.user_agent = (user_agent or os.getenv("STOCKANALYSIS_USER_AGENT", "")).strip() or "Mozilla/5.0"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_fetch_failures: list[dict[str, str]] = []

    def fetch_daily_prices(self, tickers: Iterable[str], start_date: str, end_date: str) -> pl.DataFrame:
        requested = tuple(sorted({ticker.strip().upper() for ticker in tickers if ticker and ticker.strip()}))
        if not requested:
            self.last_fetch_failures = []
            return _empty_prices()

        self.last_fetch_failures = []
        frames: list[pl.DataFrame] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(requested))) as executor:
            futures = {
                executor.submit(self._fetch_single_ticker, ticker, start_date, end_date): ticker
                for ticker in requested
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    frame = future.result()
                except Exception as exc:
                    self.last_fetch_failures.append({"ticker": ticker, "error": str(exc)})
                    continue
                if not frame.is_empty():
                    frames.append(frame)
        if not frames:
            return _empty_prices()
        return pl.concat(frames, how="vertical").sort(["ticker", "date"])

    def _fetch_single_ticker(self, ticker: str, start_date: str, end_date: str) -> pl.DataFrame:
        payload = self._load_or_fetch_payload(ticker)
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            return _empty_prices()
        frame = pl.DataFrame(rows).select(
            [
                pl.col("t").cast(pl.Utf8).alias("date"),
                pl.col("o").cast(pl.Float64, strict=False).alias("open"),
                pl.col("h").cast(pl.Float64, strict=False).alias("high"),
                pl.col("l").cast(pl.Float64, strict=False).alias("low"),
                pl.col("c").cast(pl.Float64, strict=False).alias("close"),
                pl.col("a").cast(pl.Float64, strict=False).alias("adjusted_close"),
                pl.col("v").cast(pl.Float64, strict=False).alias("volume"),
            ]
        )
        if frame.is_empty():
            return _empty_prices()
        return (
            frame.filter(pl.col("date").is_between(pl.lit(start_date), pl.lit(end_date)))
            .with_columns(pl.lit(f"{ticker}.US").alias("ticker"))
            .select(["date", "open", "high", "low", "close", "volume", "adjusted_close", "ticker"])
            .sort(["ticker", "date"])
        )

    def _load_or_fetch_payload(self, ticker: str) -> dict[str, object]:
        cache_path = self.cache_dir / f"{ticker}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cache_path.unlink(missing_ok=True)

        last_error: Exception | None = None
        for candidate in _candidate_symbols(ticker):
            url = f"{API_BASE_URL}/symbol/s/{candidate}/history?range=Max&period=Daily"
            response = requests.get(
                url,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
                timeout=self.timeout_seconds,
            )
            if response.status_code != 200:
                last_error = RuntimeError(f"{response.status_code} from {url}")
                continue
            payload = response.json()
            if payload.get("status") != 200 or not isinstance(payload.get("data"), list):
                last_error = RuntimeError(f"unexpected payload for {ticker}: {payload.get('status')}")
                continue
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"unable to fetch StockAnalysis history for {ticker}")


def _candidate_symbols(ticker: str) -> tuple[str, ...]:
    variants = [ticker, ticker.replace(".", "-"), ticker.replace("-", ".")]
    ordered: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        cleaned = variant.strip().upper()
        if cleaned and cleaned not in seen:
            ordered.append(cleaned)
            seen.add(cleaned)
    return tuple(ordered)


def _empty_prices() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "date": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "adjusted_close": pl.Float64,
            "ticker": pl.String,
        }
    )


def _load_local_dotenv() -> None:
    cwd = Path.cwd()
    for path in (cwd / ".env", cwd / ".env.local"):
        if path.exists():
            load_dotenv(path, override=False)
