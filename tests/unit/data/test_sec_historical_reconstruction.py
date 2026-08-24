from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from alpharank.data.sources.sec_historical import (
    HistoricalSecReconstructionConfig,
    identity_name_similarity,
    reconstruct_historical_sec_companyfacts,
)


class FakeCompanyFactsClient:
    def __init__(self, *, must_fail: bool = False) -> None:
        self.must_fail = must_fail

    def fetch_company_facts(self, cik: str) -> dict[str, Any]:
        if self.must_fail:
            raise RuntimeError(f"Companyfacts unavailable for {cik}")
        return {"entityName": "Example Holdings Incorporated"}

    def extract_financials(self, ticker: str, cik: str) -> pl.DataFrame:
        del cik
        return _financials(ticker, source="sec_companyfacts", dates=("2019-12-31", "2021-12-31"))


class FakeFilingClient:
    def fetch_company_submissions(self, cik: str) -> dict[str, Any]:
        del cik
        return {"name": "Example Holdings Inc", "formerNames": []}

    def extract_financials(self, ticker: str, cik: str, year: int) -> pl.DataFrame:
        del cik
        if year != 2020:
            return _financials(ticker, source="sec_filing", dates=())
        return _financials(ticker, source="sec_filing", dates=("2020-12-31",))


def test_identity_name_similarity_ignores_legal_suffixes() -> None:
    assert (
        identity_name_similarity(
            "Example Holdings Corp",
            ["Example Holdings Incorporated"],
        )
        == 1.0
    )


def test_reconstruction_writes_quarantined_companyfacts_candidate(tmp_path: Path) -> None:
    bridge_path = _write_bridge(tmp_path)
    output_dir = tmp_path / "candidate"
    production_pointer = tmp_path / "data" / "model_inputs" / "manifests" / "latest.json"
    production_pointer.parent.mkdir(parents=True)
    production_pointer.write_text('{"snapshot_id": "frozen"}\n', encoding="utf-8")

    manifest_path = reconstruct_historical_sec_companyfacts(
        _config(output_dir, bridge_path),
        companyfacts_client=FakeCompanyFactsClient(),
        filing_client=FakeFilingClient(),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    financials = pl.read_parquet(output_dir / "financials_sec_companyfacts.parquet")
    assert manifest["promotion_status"].startswith("blocked_")
    assert manifest["bridge"]["sha256"]
    assert manifest["artifacts"]["financials"]["path"] == "financials_sec_companyfacts.parquet"
    assert financials["date"].to_list() == ["2019-12-31"]
    assert production_pointer.read_text(encoding="utf-8") == '{"snapshot_id": "frozen"}\n'


def test_reconstruction_uses_auditable_filing_fallback(tmp_path: Path) -> None:
    bridge_path = _write_bridge(tmp_path)
    output_dir = tmp_path / "candidate"

    reconstruct_historical_sec_companyfacts(
        _config(output_dir, bridge_path),
        companyfacts_client=FakeCompanyFactsClient(must_fail=True),
        filing_client=FakeFilingClient(),
    )

    audit = pl.read_csv(output_dir / "historical_sec_mapping_audit.csv")
    financials = pl.read_parquet(output_dir / "financials_sec_companyfacts.parquet")
    assert audit["source_mode"].to_list() == ["sec_filing_fallback"]
    assert "Companyfacts unavailable" in audit["errors"].item()
    assert financials["source"].to_list() == ["sec_filing"]


def _config(output_dir: Path, bridge_path: Path) -> HistoricalSecReconstructionConfig:
    return HistoricalSecReconstructionConfig(
        output_dir=output_dir,
        bridge_path=bridge_path,
        retrieved_at=datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc),
        workers=1,
    )


def _write_bridge(tmp_path: Path) -> Path:
    bridge_path = tmp_path / "bridge.csv"
    bridge_path.write_text(
        "ticker,name,exchange,cik,start_date,end_date,mapping_source,mapping_priority\n"
        "OLD,Example Holdings Corp,NYSE,0000000001,2019-01-01,2020-12-31,"
        "sec_manual_historical_bridge,0\n",
        encoding="utf-8",
    )
    return bridge_path


def _financials(ticker: str, *, source: str, dates: tuple[str, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "ticker": f"{ticker}.US",
                "statement": "income_statement",
                "metric": "revenue",
                "date": period_end,
                "filing_date": period_end,
                "value": 1.0,
                "source": source,
                "source_label": "Revenue",
                "accession_number": "accession",
                "form": "10-K",
                "fiscal_period": "FY",
                "fiscal_year": int(period_end[:4]),
            }
            for period_end in dates
        ],
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "filing_date": pl.String,
            "value": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "accession_number": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
        },
    )
