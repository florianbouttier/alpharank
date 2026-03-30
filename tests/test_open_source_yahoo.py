from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpharank.data.open_source.yahoo import YahooFinanceClient, _normalize_yahoo_symbol


class _FakeTicker:
    def __init__(self, history: pd.DataFrame | None = None, exc: Exception | None = None) -> None:
        self._history = history
        self._exc = exc

    def get_earnings_dates(self, *, limit: int) -> pd.DataFrame | None:
        if self._exc is not None:
            raise self._exc
        return self._history


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


def test_normalize_yahoo_symbol_rewrites_dot_share_classes() -> None:
    assert _normalize_yahoo_symbol("BRK.B") == "BRK-B"
    assert _normalize_yahoo_symbol("BF.B") == "BF-B"
    assert _normalize_yahoo_symbol("AAPL") == "AAPL"
