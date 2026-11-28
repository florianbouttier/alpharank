from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from alpharank.data.ingestion.orchestration import run_open_source_ingestion
from alpharank.data.ingestion.raw_archive_frames import file_sha256
from alpharank.data.ingestion.sec_cumulative import (
    prepare_cumulative_sec_frames,
    resolve_sec_cumulative_parent,
)
from alpharank.data.ingestion.storage import read_json, write_json
from alpharank.data.ingestion.transaction import OpenSourceStoreTransaction


def _facts(filing_date: str, period_end: str, value: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "statement": ["income_statement"],
            "metric": ["net_income"],
            "date": [period_end],
            "filing_date": [filing_date],
            "source": ["sec_companyfacts"],
            "value": [value],
            "ingested_at": [f"{filing_date}T09:00:00Z"],
        }
    )


def _parent_manifest(path: Path) -> Path:
    path.mkdir()
    retained = _facts("2026-08-01", "2026-06-30", 10.0)
    earnings = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "period_end": ["2026-06-30"],
            "reportDate": ["2026-08-01"],
            "accession_number": ["one"],
            "source": ["sec_companyfacts"],
            "ingested_at": ["2026-08-01T09:00:00Z"],
        }
    )
    outputs = {}
    for name, frame in {
        "financials_sec_companyfacts.parquet": retained,
        "financials_sec_filing.parquet": retained.clear(),
        "earnings_sec_calendar.parquet": earnings,
        "earnings_sec_actuals.parquet": earnings,
        "general_reference_lineage.parquet": pl.DataFrame(
            {"ticker": ["AAPL.US"], "sec_cik": ["0000320193"], "source": ["sec_only_general"]}
        ),
    }.items():
        dataset = path / name
        frame.write_parquet(dataset)
        outputs[name] = {"path": str(dataset), "sha256": file_sha256(dataset)}
    manifest = path / "version_rebuild_manifest.json"
    write_json(manifest, {"cumulative_raw_outputs": outputs})
    return manifest


def test_parent_hash_and_knowledge_cutoff_fail_closed(tmp_path: Path) -> None:
    manifest = _parent_manifest(tmp_path / "parent")
    with pytest.raises(RuntimeError, match="later than.*2026-07-31"):
        resolve_sec_cumulative_parent(manifest, cutoff="2026-07-31")

    (manifest.parent / "financials_sec_companyfacts.parquet").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch.*financials_sec_companyfacts"):
        resolve_sec_cumulative_parent(manifest, cutoff="2026-08-01")


def test_ingestion_refuses_invalid_parent_before_creating_a_transaction(tmp_path: Path) -> None:
    manifest = _parent_manifest(tmp_path / "parent")
    official = tmp_path / "open_source" / "official"
    with pytest.raises(RuntimeError, match="later than.*2026-07-31"):
        run_open_source_ingestion(
            live_dir=official,
            end_date="2026-07-31",
            sec_cumulative_parent_manifest=manifest,
        )
    assert not official.parent.exists()


def test_cumulative_sec_uses_verified_parent_and_retains_evidence_on_rollback(
    tmp_path: Path,
) -> None:
    manifest = _parent_manifest(tmp_path / "parent")
    parent = resolve_sec_cumulative_parent(manifest, cutoff="2026-11-20")
    assert parent is not None
    official = tmp_path / "open_source" / "official"
    live_raw = official / "raw"
    live_raw.mkdir(parents=True)
    stale = live_raw / "financials_sec_companyfacts.parquet"
    stale.write_bytes(b"stale live state")
    run = official / "runs" / "20261120_080000"
    current = _facts("2026-11-03", "2026-09-30", 20.0)
    changed = _facts("2026-08-01", "2026-06-30", 11.0).with_columns(
        pl.lit("2026-11-20T09:00:00Z").alias("ingested_at")
    )
    calendar = pl.read_parquet(manifest.parent / "earnings_sec_calendar.parquet").clear()

    with pytest.raises(RuntimeError, match="historical revision gate"):
        with OpenSourceStoreTransaction(official_dir=official):
            frames = prepare_cumulative_sec_frames(
                parent=parent,
                run_directory=run,
                companyfacts=pl.concat([current, changed]),
                filing=current.clear(),
                calendar=calendar,
                actuals=calendar,
                issuer_mapping=pl.DataFrame({"ticker": ["AAPL"], "cik": ["320193"]}),
            )
            assert frames.companyfacts.select("filing_date", "value").rows() == [
                ("2026-08-01", 11.0),
                ("2026-11-03", 20.0),
            ]
            assert frames.calendar.height == 1
            stale.write_bytes(b"uncommitted state")
            raise RuntimeError("historical revision gate")

    assert stale.read_bytes() == b"stale live state"
    assert read_json(run / "sec_cumulative" / "version_rebuild_manifest.json")["parent"][
        "manifest_sha256"
    ] == file_sha256(manifest)
    assert frames.receipt["manifest_path"] == str(
        run / "sec_cumulative" / "version_rebuild_manifest.json"
    )


def test_acquisition_issuer_change_is_retained_and_rejected(tmp_path: Path) -> None:
    manifest = _parent_manifest(tmp_path / "parent")
    parent = resolve_sec_cumulative_parent(manifest, cutoff="2026-11-20")
    assert parent is not None
    run = tmp_path / "run"
    facts = _facts("2026-11-03", "2026-09-30", 20.0)
    calendar = pl.read_parquet(manifest.parent / "earnings_sec_calendar.parquet").clear()

    with pytest.raises(RuntimeError, match="disagree on issuer CIK.*AAPL.US"):
        prepare_cumulative_sec_frames(
            parent=parent,
            run_directory=run,
            companyfacts=facts,
            filing=facts.clear(),
            calendar=calendar,
            actuals=calendar,
            issuer_mapping=pl.DataFrame({"ticker": ["AAPL"], "cik": ["320194"]}),
        )

    assert (run / "sec_cumulative_issuer_map.parquet").is_file()
    assert (run / "raw" / "financials_sec_companyfacts.parquet").is_file()
    assert not (run / "sec_cumulative").exists()
