from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from alpharank.data.open_source.yahoo import YahooFinanceClient, _normalize_yahoo_symbol


class _FakeTicker:
    def __init__(self, history: pd.DataFrame | None = None, exc: Exception | None = None) -> None:
        self._history = history
        self._exc = exc

    def get_earnings_dates(self, *, limit: int) -> pd.DataFrame | None:
        if self._exc is not None:
            raise self._exc
        return self._history


def test_stock_split_fetch_retries_with_fresh_ticker(monkeypatch) -> None:
    empty_actions = pd.DataFrame()
    split_actions = pd.DataFrame(
        {"Stock Splits": [2.0]},
        index=pd.to_datetime(["2026-08-11"]),
    )

    class FakeSplitTicker:
        def __init__(self, actions: pd.DataFrame) -> None:
            self.actions = actions

    client = YahooFinanceClient()
    monkeypatch.setattr(client, "_ticker", lambda ticker: FakeSplitTicker(empty_actions))
    monkeypatch.setattr(client, "_fresh_ticker", lambda ticker: FakeSplitTicker(split_actions))
    monkeypatch.setattr("alpharank.data.open_source.yahoo.time.sleep", lambda seconds: None)

    splits = client.fetch_stock_splits(["MNST"])

    assert splits.select("ticker", "date", "split_ratio").rows() == [
        ("MNST.US", "2026-08-11", 2.0)
    ]


def test_stock_split_fetch_reuses_event_from_price_download(monkeypatch) -> None:
    columns = pd.MultiIndex.from_product(
        [["MNST"], ["Close", "Stock Splits"]]
    )
    history = pd.DataFrame(
        [[45.53, 2.0]],
        index=pd.to_datetime(["2026-08-11"]),
        columns=columns,
    )
    client = YahooFinanceClient()
    client._record_downloaded_splits(
        history,
        request_symbol="MNST",
        ticker="MNST",
    )
    monkeypatch.setattr(
        client,
        "_fetch_actions_with_retries",
        lambda ticker: (_ for _ in ()).throw(AssertionError("secondary endpoint called")),
    )

    splits = client.fetch_stock_splits(["MNST"])

    assert splits.to_dicts() == [
        {
            "ticker": "MNST.US",
            "date": "2026-08-11",
            "split_ratio": 2.0,
            "source": "yahoo_price_download_actions",
        }
    ]


def test_fetch_earnings_dates_skips_ticker_errors(tmp_path: Path) -> None:
    client = YahooFinanceClient(cache_dir=tmp_path / "cache")
    good_history = pd.DataFrame(
        {
            "EPS Estimate": [1.0],
            "Reported EPS": [1.1],
            "Surprise(%)": [10.0],
        },
        index=pd.Index([pd.Timestamp("2025-01-30 21:00:00")], name="Earnings Date"),
    )
    fake_tickers = {
        "AAPL": _FakeTicker(history=good_history),
        "BROKEN": _FakeTicker(exc=KeyError(["Earnings Date"])),
    }
    client._ticker = lambda symbol: fake_tickers[symbol]  # type: ignore[method-assign]

    result = client.fetch_earnings_dates(["AAPL", "BROKEN"])

    assert result.height == 1
    assert result["ticker"].to_list() == ["AAPL.US"]


def test_fetch_earnings_dates_retries_with_fresh_ticker(tmp_path: Path) -> None:
    client = YahooFinanceClient(cache_dir=tmp_path / "cache")
    good_history = pd.DataFrame(
        {
            "EPS Estimate": [2.0],
            "Reported EPS": [2.5],
            "Surprise(%)": [25.0],
        },
        index=pd.Index([pd.Timestamp("2026-02-19 16:00:00")], name="Earnings Date"),
    )
    client._ticker = lambda symbol: _FakeTicker(history=None)  # type: ignore[method-assign]
    client._fresh_ticker = lambda symbol: _FakeTicker(history=good_history)  # type: ignore[method-assign]

    result = client.fetch_earnings_dates(["NEM"], limit=100)

    assert result.height == 1
    assert result["ticker"].to_list() == ["NEM.US"]
    assert result["epsActual"].to_list() == [2.5]


def test_normalize_yahoo_symbol_rewrites_dot_share_classes() -> None:
    assert _normalize_yahoo_symbol("BRK.B") == "BRK-B"
    assert _normalize_yahoo_symbol("BF.B") == "BF-B"
    assert _normalize_yahoo_symbol("AAPL") == "AAPL"


def test_quarterly_financials_are_fetched_once_per_run(tmp_path: Path, monkeypatch) -> None:
    client = YahooFinanceClient(cache_dir=tmp_path / "cache")
    calls: list[str] = []

    def fetch(ticker: str) -> list[pl.DataFrame]:
        calls.append(ticker)
        return [
            pl.DataFrame(
                {
                    "ticker": [f"{ticker}.US"],
                    "statement": ["income_statement"],
                    "metric": ["revenue"],
                    "date": ["2026-03-31"],
                    "filing_date": [None],
                    "value": [1.0],
                    "source": ["yfinance"],
                    "source_label": ["Total Revenue"],
                }
            )
        ]

    monkeypatch.setattr("alpharank.data.open_source.yahoo._fetch_ticker_financial_frames", fetch)

    assert client.fetch_quarterly_financials(["AAPL"]).height == 1
    assert client.fetch_quarterly_financials(["AAPL.US"]).height == 1
    assert calls == ["AAPL"]
