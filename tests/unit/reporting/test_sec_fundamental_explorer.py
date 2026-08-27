from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from alpharank.reporting.sec_fundamental_explorer import (
    SecExplorerConfig,
    build_sec_fundamental_explorer,
)


def test_sec_explorer_preserves_every_downloaded_version_in_offline_html(
    tmp_path: Path,
) -> None:
    raw_dir = _write_raw_run(tmp_path, run_id="20260827_070654")
    result = build_sec_fundamental_explorer(_config(tmp_path=tmp_path, raw_dir=raw_dir))

    html = result.report_path.read_text(encoding="utf-8")
    payload = _decode_payload(html)
    companyfacts = payload["datasets"]["companyfacts"]
    ticker_rows = companyfacts["rows_by_ticker"]["ABC.US"]
    columns = companyfacts["columns"]
    filing_date_index = columns.index("filing_date")
    metric_index = columns.index("metric")
    revenue_rows = [row for row in ticker_rows if row[metric_index] == "revenue"]

    assert result.company_count == 2
    assert result.sec_row_count == 9
    assert len(ticker_rows) == 3
    assert [row[filing_date_index] for row in revenue_rows] == [
        "2024-02-01",
        "2024-03-01",
    ]
    assert "DecompressionStream" in html
    assert "Toutes les versions" in html
    assert "Tout ce qui a été téléchargé" in html
    assert "https://" not in html
    assert "http://" not in html
    assert re.search(r"<script[^>]+src=", html) is None


def test_sec_explorer_manifest_hashes_report_payload_and_source_files(tmp_path: Path) -> None:
    raw_dir = _write_raw_run(tmp_path, run_id="20260827_070654")
    result = build_sec_fundamental_explorer(_config(tmp_path=tmp_path, raw_dir=raw_dir))

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    report_hash = hashlib.sha256(result.report_path.read_bytes()).hexdigest()
    source_paths = {item["dataset"]: item["path"] for item in manifest["source_files"]}

    assert manifest["run_id"] == "20260827_070654"
    assert manifest["report"]["sha256"] == report_hash == result.report_sha256
    assert manifest["report"]["external_assets"] == 0
    assert manifest["coverage"] == {
        "company_count": 2,
        "sec_row_count": 9,
        "source_file_count": 6,
    }
    assert source_paths["companyfacts"].startswith("data/open_source/official/runs/")
    assert all(not path.startswith("/") for path in source_paths.values())


def test_sec_explorer_rejects_rows_from_another_ingestion_run(tmp_path: Path) -> None:
    raw_dir = _write_raw_run(tmp_path, run_id="20260827_070654")
    companyfacts_path = raw_dir / "financials_sec_companyfacts.parquet"
    companyfacts = pl.read_parquet(companyfacts_path).with_columns(
        pl.lit("another_run").alias("ingestion_run_id")
    )
    companyfacts.write_parquet(companyfacts_path)

    with pytest.raises(ValueError, match="not sealed to run 20260827_070654"):
        build_sec_fundamental_explorer(_config(tmp_path=tmp_path, raw_dir=raw_dir))


def _config(*, tmp_path: Path, raw_dir: Path) -> SecExplorerConfig:
    return SecExplorerConfig(
        raw_run_dir=raw_dir,
        output_dir=tmp_path / "outputs" / "sec_fundamental_explorer" / raw_dir.parent.name,
        project_root=tmp_path,
        generated_at_utc=datetime(2026, 8, 27, 20, 0, tzinfo=timezone.utc),
        initial_ticker="ABC.US",
    )


def _decode_payload(html: str) -> dict[str, object]:
    match = re.search(
        r'<script id="sec-explorer-payload"[^>]*>([A-Za-z0-9+/=]+)</script>',
        html,
    )
    if match is None:
        raise AssertionError("Embedded SEC payload is missing")
    return json.loads(gzip.decompress(base64.b64decode(match.group(1))))


def _write_raw_run(tmp_path: Path, *, run_id: str) -> Path:
    raw_dir = tmp_path / "data" / "open_source" / "official" / "runs" / run_id / "raw"
    raw_dir.mkdir(parents=True)
    _financial_rows(run_id).write_parquet(raw_dir / "financials_sec_companyfacts.parquet")
    _filing_rows(run_id).write_parquet(raw_dir / "financials_sec_filing.parquet")
    _calendar_rows(run_id).write_parquet(raw_dir / "earnings_sec_calendar.parquet")
    _actual_rows(run_id).write_parquet(raw_dir / "earnings_sec_actuals.parquet")
    _reference_rows(run_id).write_parquet(raw_dir / "general_reference.parquet")
    _reference_lineage_rows(run_id).write_parquet(raw_dir / "general_reference_lineage.parquet")
    (raw_dir.parent / "acquisition_status.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "sources": [
                    {
                        "source": "sec_companyfacts",
                        "status": "downloaded_with_failures",
                        "downloaded_rows": 4,
                        "failure_count": 1,
                        "failure_examples": [{"ticker": "OLD", "error": "404"}],
                    },
                    {
                        "source": "sec_submissions",
                        "status": "downloaded",
                        "downloaded_rows": 3,
                        "failure_count": 0,
                    },
                    {
                        "source": "sec_filing_documents",
                        "status": "downloaded",
                        "downloaded_rows": 1,
                        "failure_count": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir.parent / "source_refresh_contract.json").write_text(
        json.dumps(
            {
                "source_semantics": {
                    "sec_companyfacts": {"fetch": "network_full_company_payload"},
                    "sec_submissions": {"fetch": "network_full_company_payload"},
                    "sec_filing_documents": {"fetch": "network_on_demand"},
                }
            }
        ),
        encoding="utf-8",
    )
    return raw_dir


def _financial_rows(run_id: str) -> pl.DataFrame:
    shared = {
        "ticker": ["ABC.US", "ABC.US", "ABC.US", "XYZ.US"],
        "statement": ["income_statement"] * 4,
        "metric": ["revenue", "revenue", "net_income", "revenue"],
        "date": ["2023-12-31"] * 3 + ["2023-09-30"],
        "filing_date": ["2024-02-01", "2024-03-01", "2024-02-01", "2023-11-01"],
        "value": [100.0, 101.0, 12.0, 50.0],
        "source": ["sec_companyfacts"] * 4,
        "source_label": ["Revenue", "Revenue", "NetIncome", "Revenue"],
        "accession_number": [None] * 4,
        "form": ["10-K", "10-K/A", "10-K", "10-Q"],
        "fiscal_period": ["Q4", "Q4", "Q4", "Q3"],
        "fiscal_year": [2023] * 4,
        "dataset": ["financials_sec_companyfacts"] * 4,
        "ingestion_run_id": [run_id] * 4,
        "ingested_at": ["2026-08-27T20:00:00+00:00"] * 4,
    }
    return pl.DataFrame(shared)


def _filing_rows(run_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["ABC.US"],
            "statement": ["balance_sheet"],
            "metric": ["total_assets"],
            "date": ["2023-12-31"],
            "filing_date": ["2024-02-01"],
            "value": [400.0],
            "source": ["sec_filing"],
            "source_label": ["Assets"],
            "accession_number": ["0001"],
            "form": ["10-K"],
            "fiscal_period": ["Q4"],
            "fiscal_year": [2023],
            "dataset": ["financials_sec_filing"],
            "ingestion_run_id": [run_id],
            "ingested_at": ["2026-08-27T20:00:00+00:00"],
        }
    )


def _calendar_rows(run_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["ABC.US", "ABC.US", "XYZ.US"],
            "period_end": ["2023-12-31", "2023-12-31", "2023-09-30"],
            "reportDate": ["2024-02-01", "2024-03-01", "2023-11-01"],
            "earningsDatetime": [None, None, None],
            "epsEstimate": [None, None, None],
            "epsActual": [None, None, None],
            "surprisePercent": [None, None, None],
            "source": ["sec_submissions"] * 3,
            "source_label": ["reportDate"] * 3,
            "calendar_source": [None, None, None],
            "actual_source": [None, None, None],
            "estimate_source": [None, None, None],
            "accession_number": ["0001", "0002", "0003"],
            "form": ["10-K", "10-K/A", "10-Q"],
            "fiscal_period": ["Q4", "Q4", "Q3"],
            "fiscal_year": [2023] * 3,
            "dataset": ["earnings_sec_calendar"] * 3,
            "ingestion_run_id": [run_id] * 3,
            "ingested_at": ["2026-08-27T20:00:00+00:00"] * 3,
        }
    )


def _actual_rows(run_id: str) -> pl.DataFrame:
    frame = (
        _calendar_rows(run_id)
        .head(1)
        .with_columns(
            pl.lit(1.25).alias("epsActual"),
            pl.lit("EarningsPerShareDiluted").alias("source_label"),
            pl.lit("sec_companyfacts").alias("source"),
            pl.lit("earnings_sec_actuals").alias("dataset"),
        )
    )
    return frame


def _reference_rows(run_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["ABC.US", "XYZ.US"],
            "name": ["ABC Corp", "XYZ Corp"],
            "exchange": ["NYSE", "NASDAQ"],
            "cik": ["0000000001", "0000000002"],
            "source": ["open_source_general"] * 2,
            "Sector": ["Industrials", "Technology"],
            "industry": ["Tools", "Software"],
            "sector_source": ["yfinance"] * 2,
            "sector_raw_value": ["Industrials", "Technology"],
            "sic": ["1000", "2000"],
            "sic_description": ["Tools", "Software"],
            "mapping_rule": ["yfinance:sector"] * 2,
            "dataset": ["general_reference"] * 2,
            "ingestion_run_id": [run_id] * 2,
            "ingested_at": ["2026-08-27T20:00:00+00:00"] * 2,
        }
    )


def _reference_lineage_rows(run_id: str) -> pl.DataFrame:
    base = _reference_rows(run_id)
    return base.with_columns(
        pl.col("name").str.to_uppercase().alias("sec_name"),
        pl.col("exchange").alias("sec_exchange"),
        pl.col("cik").alias("sec_cik"),
        pl.col("sic").alias("sec_sic"),
        pl.col("sic_description").alias("sec_sic_description"),
        pl.col("name").alias("yahoo_name"),
        pl.col("exchange").alias("yahoo_exchange"),
        pl.col("Sector").alias("yahoo_sector"),
        pl.col("industry").alias("yahoo_industry"),
        pl.lit("sec_mapping").alias("selected_name_source"),
        pl.lit("sec_mapping").alias("selected_exchange_source"),
        pl.lit("general_reference_lineage").alias("dataset"),
    )
