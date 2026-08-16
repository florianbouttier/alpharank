from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from alpharank.data.open_source.refresh_policy import PRODUCTION_SOURCE_REFRESH_POLICY
from alpharank.data.open_source.ingestion import (
    _drop_refreshed_partitions,
    _fetch_sec_companyfacts_bundle,
    _required_failure_tickers,
    _resolve_sec_mapping_coverage,
)
from alpharank.data.open_source.sec import SecCompanyFactsClient
from alpharank.data.open_source.sec_filing import SecFilingFactsClient
from alpharank.data.open_source.stockanalysis import StockAnalysisClient


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def test_production_refresh_contract_is_full_and_does_not_persist_sec_payloads() -> None:
    contract = PRODUCTION_SOURCE_REFRESH_POLICY.to_manifest(
        mode="daily",
        price_start_date="2005-01-01",
        price_end_date="2026-08-13",
        financial_years=tuple(range(2005, 2027)),
    )

    assert contract["snapshot_scope"] == "full_ingestion"
    assert contract["source_semantics"]["yfinance_prices"]["history"].startswith("full available active-universe history")
    assert contract["source_semantics"]["sec_companyfacts"]["persistent_payload_cache"] is False
    assert contract["source_semantics"]["sec_filing_documents"]["persistent_cache"] is False
    assert contract["policy"]["require_eodhd_price_seed"] is True
    assert contract["policy"]["historical_price_return_revision_threshold"] == 0.0001
    assert contract["policy"]["allow_historical_price_revisions"] is False


def test_companyfacts_refresh_bypasses_disk_cache_but_reuses_run_memory(tmp_path: Path) -> None:
    cache_dir = tmp_path / "companyfacts"
    cache_dir.mkdir()
    cache_path = cache_dir / "payload.json"
    cache_path.write_text('{"version": "stale"}', encoding="utf-8")
    calls = 0

    client = SecCompanyFactsClient(
        user_agent="test",
        cache_dir=cache_dir,
        refresh_cache=True,
        persist_cache=False,
        request_pause_seconds=0,
    )

    def get(url: str, timeout: int) -> _Response:
        nonlocal calls
        calls += 1
        return _Response({"version": "fresh"})

    client.session.get = get  # type: ignore[method-assign]
    assert client._get_json("https://example.test/payload", "payload.json") == {"version": "fresh"}
    assert client._get_json("https://example.test/payload", "payload.json") == {"version": "fresh"}
    assert calls == 1
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {"version": "stale"}

    company_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000123.json"
    client._response_cache[company_url] = {"version": "transient"}
    client.discard_company_facts("123")
    assert company_url not in client._response_cache


def test_sec_filing_refresh_does_not_materialize_source_payload(tmp_path: Path) -> None:
    cache_dir = tmp_path / "filings"
    client = SecFilingFactsClient(
        user_agent="test",
        cache_dir=cache_dir,
        refresh_mutable_cache=True,
        persist_metadata_cache=False,
        persist_filing_documents=False,
        request_pause_seconds=0,
    )
    client.session.get = lambda url, timeout: _Response({"fresh": True})  # type: ignore[method-assign]

    assert client._get_json("https://example.test/submissions", "submissions.json", refresh=True) == {"fresh": True}
    assert client._get_text(
        "https://example.test/filing.xml",
        "filing.xml",
        refresh=True,
        persist=False,
    ) == '{"fresh": true}'
    assert list(cache_dir.iterdir()) == []


def test_stockanalysis_refresh_ignores_and_does_not_replace_disk_cache(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "stockanalysis"
    cache_dir.mkdir()
    cache_path = cache_dir / "AAPL.json"
    cache_path.write_text('{"status": 200, "data": [{"t": "stale"}]}', encoding="utf-8")
    client = StockAnalysisClient(cache_dir=cache_dir, refresh_cache=True, persist_cache=False)

    monkeypatch.setattr(
        "alpharank.data.open_source.stockanalysis.requests.get",
        lambda *args, **kwargs: _Response({"status": 200, "data": [{"t": "fresh"}]}),
    )

    assert client._load_or_fetch_payload("AAPL")["data"] == [{"t": "fresh"}]
    assert json.loads(cache_path.read_text(encoding="utf-8"))["data"] == [{"t": "stale"}]


def test_required_failure_tickers_ignores_inactive_history_failures() -> None:
    failures = [
        {"ticker": "AAPL", "error": "temporary"},
        {"ticker": "OLD.US", "error": "delisted"},
    ]

    assert _required_failure_tickers(
        failures,
        required_tickers=("AAPL.US", "MSFT"),
    ) == ("AAPL",)


def test_companyfacts_bundle_discards_each_company_payload(monkeypatch) -> None:
    class Client:
        def __init__(self) -> None:
            self.discarded: list[str] = []

        def extract_financials(self, ticker: str, cik: str) -> pl.DataFrame:
            return pl.DataFrame({"ticker": [ticker], "cik": [cik]})

        def discard_company_facts(self, cik: str) -> None:
            self.discarded.append(cik)

    client = Client()
    monkeypatch.setattr(
        "alpharank.data.open_source.ingestion._extract_sec_companyfacts_earnings_actuals",
        lambda sec_client, ticker, cik: pl.DataFrame({"ticker": [ticker], "cik": [cik]}),
    )
    mapping = pl.DataFrame({"ticker": ["AAPL", "MSFT"], "cik": ["1", "2"]})

    financials, earnings, failures = _fetch_sec_companyfacts_bundle(client, mapping)

    assert len(financials) == len(earnings) == 2
    assert failures == []
    assert client.discarded == ["1", "2"]


def test_full_refresh_partition_replacement_retains_unrefreshed_history(tmp_path: Path) -> None:
    path = tmp_path / "raw.parquet"
    pl.DataFrame(
        {
            "ticker": ["AAPL.US", "AAPL.US", "OLD.US"],
            "date": ["2004-12-31", "2020-03-31", "2020-03-31"],
            "value": [1.0, 2.0, 3.0],
        }
    ).write_parquet(path)

    _drop_refreshed_partitions(
        path,
        tickers=("AAPL",),
        date_column="date",
        start_date="2005-01-01",
        end_date="2026-08-13",
    )

    retained = pl.read_parquet(path).sort(["ticker", "date"])
    assert retained.select(["ticker", "date"]).rows() == [
        ("AAPL.US", "2004-12-31"),
        ("OLD.US", "2020-03-31"),
    ]


def test_sec_mapping_coverage_normalizes_ticker_suffixes() -> None:
    mapping = pl.DataFrame({"ticker": ["AAPL.US", "MSFT"]})

    mapped, required, missing = _resolve_sec_mapping_coverage(
        sec_mapping=mapping,
        required_tickers=("AAPL", "MSFT.US", "NVDA"),
    )

    assert mapped == {"AAPL", "MSFT"}
    assert required == {"AAPL", "MSFT", "NVDA"}
    assert missing == ("NVDA",)
