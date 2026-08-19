from __future__ import annotations

import polars as pl

from alpharank.data.open_source.ingestion import (
    _confirmed_terminal_price_tickers,
    _complete_yahoo_history_against_validated,
    _consolidate_price_sources,
    _download_yahoo_price_history,
    _identify_simfin_price_fallback_tickers,
    _identify_stockanalysis_price_fallback_tickers,
    _network_price_refresh_coverage,
    _with_price_ingestion_metadata,
)


def _price_frame(*, ticker: str, adjusted_close: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2025-01-02"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1_000.0],
            "adjusted_close": [adjusted_close],
            "ticker": [ticker],
        }
    )


def test_price_consolidation_prefers_yfinance_and_keeps_simfin_gap_fill() -> None:
    yahoo = _with_price_ingestion_metadata(
        _price_frame(ticker="IPG.US", adjusted_close=100.0),
        dataset="prices_yfinance",
        run_id="run",
        ingested_at="2026-04-01T00:00:00Z",
    )
    simfin = pl.concat(
        [
            _with_price_ingestion_metadata(
                _price_frame(ticker="IPG.US", adjusted_close=99.5),
                dataset="prices_simfin",
                source="simfin",
                run_id="run",
                ingested_at="2026-04-01T00:00:00Z",
            ),
            _with_price_ingestion_metadata(
                _price_frame(ticker="DFS.US", adjusted_close=50.0),
                dataset="prices_simfin",
                source="simfin",
                run_id="run",
                ingested_at="2026-04-01T00:00:00Z",
            ),
        ],
        how="vertical",
    )

    clean, lineage = _consolidate_price_sources([yahoo, simfin], ticker_list=["IPG", "DFS"])

    assert clean.sort("ticker")["ticker"].to_list() == ["DFS.US", "IPG.US"]
    assert clean.filter(pl.col("ticker") == "IPG.US")["adjusted_close"].to_list() == [100.0]
    assert lineage.filter(pl.col("ticker") == "IPG.US")["source"].to_list() == ["yfinance"]
    assert lineage.filter(pl.col("ticker") == "DFS.US")["source"].to_list() == ["simfin"]


def test_identify_simfin_price_fallback_tickers_targets_yahoo_gaps() -> None:
    yahoo = _with_price_ingestion_metadata(
        pl.concat(
            [
                _price_frame(ticker="IPG.US", adjusted_close=100.0),
                _price_frame(ticker="DFS.US", adjusted_close=50.0),
            ],
            how="vertical",
        ),
        dataset="prices_yfinance",
        run_id="run",
        ingested_at="2026-04-01T00:00:00Z",
    )

    fallback = _identify_simfin_price_fallback_tickers(
        requested_tickers=["IPG", "DFS", "K", "JNPR"],
        yahoo_prices_delta=yahoo,
        backfill_tickers=["K"],
    )

    assert fallback == ("JNPR", "K")


def test_price_consolidation_uses_stockanalysis_after_simfin() -> None:
    stockanalysis = _with_price_ingestion_metadata(
        _price_frame(ticker="K.US", adjusted_close=80.0),
        dataset="prices_stockanalysis",
        source="stockanalysis",
        run_id="run",
        ingested_at="2026-04-01T00:00:00Z",
    )
    simfin = _with_price_ingestion_metadata(
        _price_frame(ticker="K.US", adjusted_close=79.5),
        dataset="prices_simfin",
        source="simfin",
        run_id="run",
        ingested_at="2026-04-01T00:00:00Z",
    )

    clean, lineage = _consolidate_price_sources([stockanalysis, simfin], ticker_list=["K"])

    assert clean["adjusted_close"].to_list() == [79.5]
    assert lineage["source"].to_list() == ["simfin"]


def test_identify_stockanalysis_price_fallback_tickers_targets_remaining_gaps() -> None:
    covered = pl.concat(
        [
            _with_price_ingestion_metadata(
                _price_frame(ticker="IPG.US", adjusted_close=100.0),
                dataset="prices_yfinance",
                run_id="run",
                ingested_at="2026-04-01T00:00:00Z",
            ),
            _with_price_ingestion_metadata(
                _price_frame(ticker="DFS.US", adjusted_close=50.0),
                dataset="prices_simfin",
                source="simfin",
                run_id="run",
                ingested_at="2026-04-01T00:00:00Z",
            ),
        ],
        how="vertical",
    )

    fallback = _identify_stockanalysis_price_fallback_tickers(
        requested_tickers=["IPG", "DFS", "K", "JNPR"],
        covered_prices_delta=covered,
        backfill_tickers=["K"],
    )

    assert fallback == ("JNPR", "K")


def test_network_price_refresh_coverage_does_not_count_retained_rows() -> None:
    delta = _with_price_ingestion_metadata(
        _price_frame(ticker="AAPL.US", adjusted_close=100.0),
        dataset="prices_yfinance",
        run_id="run",
        ingested_at="2026-08-13T00:00:00Z",
    )

    refreshed, missing = _network_price_refresh_coverage(
        delta,
        requested_tickers=["AAPL", "MSFT"],
    )

    assert refreshed == {"AAPL"}
    assert missing == ("MSFT",)


def test_network_price_refresh_coverage_accepts_benchmark_symbol_without_suffix() -> None:
    delta = _with_price_ingestion_metadata(
        _price_frame(ticker="SPY.US", adjusted_close=100.0),
        dataset="prices_spy_yfinance",
        run_id="run",
        ingested_at="2026-08-13T00:00:00Z",
    )

    refreshed, missing = _network_price_refresh_coverage(delta, requested_tickers=["SPY"])

    assert refreshed == {"SPY"}
    assert missing == ()


def test_yahoo_history_retries_only_missing_network_tickers() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def download_prices(self, tickers, start_date, end_date):
            requested = tuple(tickers)
            self.calls.append(requested)
            ticker = "AAPL.US" if len(self.calls) == 1 else "MSFT.US"
            return _price_frame(ticker=ticker, adjusted_close=100.0)

    client = Client()
    prices = _download_yahoo_price_history(
        client,
        tickers=("AAPL", "MSFT"),
        start_date="2025-01-01",
        end_date="2026-08-13",
        max_attempts=3,
    )

    assert client.calls == [("AAPL", "MSFT"), ("MSFT",)]
    assert set(prices["ticker"].to_list()) == {"AAPL.US", "MSFT.US"}


def test_full_yahoo_vintage_repairs_missing_validated_keys() -> None:
    previous = pl.concat(
        [
            _price_frame(ticker="AAPL.US", adjusted_close=100.0),
            _price_frame(ticker="MSFT.US", adjusted_close=200.0),
        ],
        how="vertical",
    ).with_columns(pl.lit("2025-01-03").alias("date"))
    previous = pl.concat(
        [
            previous,
            previous.with_columns(pl.lit("2025-01-02").alias("date")),
        ],
        how="vertical",
    )
    initial = previous.filter(pl.col("date") == "2025-01-02")

    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def download_prices(self, tickers, start_date, end_date):
            self.calls.append(tuple(tickers))
            return previous.filter(
                pl.col("ticker").str.replace(r"\.US$", "").is_in(list(tickers))
                & (pl.col("date") == "2025-01-03")
            )

    client = Client()
    completed, report = _complete_yahoo_history_against_validated(
        client,
        initial_prices=initial,
        previous_validated_lineage=previous,
        active_tickers=("AAPL", "MSFT"),
        start_date="2025-01-01",
        end_date="2025-02-01",
    )

    assert client.calls == [("AAPL", "MSFT")]
    assert completed.height == 4
    assert report["initial_missing_key_count"] == 2
    assert report["remaining_missing_key_count"] == 0
    assert report["passed"] is True


def test_full_yahoo_vintage_retries_null_adjusted_close_key() -> None:
    previous = _price_frame(ticker="AAPL.US", adjusted_close=100.0).with_columns(
        pl.lit("2025-01-03").alias("date")
    )
    initial = previous.with_columns(pl.lit(None).cast(pl.Float64).alias("adjusted_close"))

    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def download_prices(self, tickers, start_date, end_date):
            self.calls.append(tuple(tickers))
            return previous

    client = Client()
    completed, report = _complete_yahoo_history_against_validated(
        client,
        initial_prices=initial,
        previous_validated_lineage=previous,
        active_tickers=("AAPL",),
        start_date="2025-01-01",
        end_date="2025-02-01",
    )

    assert client.calls == [("AAPL",)]
    assert completed.filter(pl.col("adjusted_close").is_not_null()).height == 1
    assert report["initial_missing_key_count"] == 1
    assert report["remaining_missing_key_count"] == 0


def test_confirmed_terminal_price_tickers_use_sourced_effective_removal(tmp_path) -> None:
    registry_path = tmp_path / "constituents.json"
    registry_path.write_text(
        """{
  "index": "S&P 500",
  "base_month": "2026-08-01",
  "events": [{
    "event_id": "sp500-20260805-ferg-ea",
    "observed_at": "2026-07-31T23:59:59-04:00",
    "effective_at": "2026-08-05T00:00:00-04:00",
    "effective_date": "2026-08-05",
    "confidence": "high",
    "source_url": "https://example.com/sourced-event",
    "operations": [
      {"action": "add", "ticker": "FERG"},
      {"action": "remove", "ticker": "EA"}
    ]
  }]
}
""",
        encoding="utf-8",
    )

    assert _confirmed_terminal_price_tickers(
        registry_path=registry_path,
        active_tickers=("EA", "AAPL"),
        expected_through="2026-08-04",
    ) == ()
    assert _confirmed_terminal_price_tickers(
        registry_path=registry_path,
        active_tickers=("EA", "AAPL"),
        expected_through="2026-08-05",
    ) == ("EA.US",)
