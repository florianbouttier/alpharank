from __future__ import annotations

import json
from datetime import datetime, timezone

import polars as pl

from alpharank.data.warehouse.sector_history import (
    build_sector_history,
    materialize_sector_history_release,
)


def _observations() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["AAA.US", "BBB.US", "CCC.US"],
            "Sector": ["Information Technology", "Health Care", "Unknown"],
            "sector_source": ["yfinance", "sec_sic", "unknown"],
            "sector_raw_value": ["Technology", "Medical devices", "Services"],
            "sic": [None, "3841", "7389"],
            "sic_description": [None, "Surgical instruments", "Business services"],
            "mapping_rule": ["yfinance:sector", "sec_sic:healthcare", "fallback:unknown"],
            "cik": ["1", "2", "3"],
            "ingestion_run_id": ["run-1", "run-1", "run-1"],
            "ingested_at": [
                "2026-01-02T12:00:00+00:00",
                "2026-01-03T12:00:00+00:00",
                "2026-01-04T12:00:00+00:00",
            ],
            "raw_observation_path": ["raw/a.parquet"] * 3,
            "raw_observation_sha256": ["a" * 64] * 3,
        }
    )


def test_sector_history_keeps_causal_lineage_and_rejects_unknown() -> None:
    result = build_sector_history(
        _observations(),
        allowed_sectors=("Technology", "Healthcare"),
        aliases={
            "Information Technology": "Technology",
            "Health Care": "Healthcare",
        },
    )

    assert result.history["ticker"].to_list() == ["AAA.US", "BBB.US"]
    assert result.history["Sector"].to_list() == ["Technology", "Healthcare"]
    assert result.history["observed_at"].to_list() == result.history["effective_at"].to_list()
    assert result.history["classification_id"].str.starts_with("sector-").all()
    assert result.rejections["ticker"].to_list() == ["CCC.US"]
    assert result.rejections["rejection_reason"].to_list() == ["sector_not_observed"]
    assert result.summary["ticker_coverage_pct"] == 200.0 / 3.0


def test_sector_history_release_is_content_addressed_and_immutable(tmp_path) -> None:
    contract = {
        "contract_id": "point_in_time_sector_history_v1",
        "schema_version": 1,
        "grain": "one observation",
        "business_key": ["ticker", "observed_at"],
        "allowed_sectors": ["Technology", "Healthcare"],
        "aliases": {
            "Information Technology": "Technology",
            "Health Care": "Healthcare",
        },
        "time_policy": {"retroactive_static_mapping_allowed": False},
        "missing_policy": {"implicit_fallback_allowed": False},
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    generated_at = datetime(2026, 1, 5, tzinfo=timezone.utc)

    first = materialize_sector_history_release(
        _observations(),
        contract_path=contract_path,
        output_root=tmp_path / "releases",
        generated_at=generated_at,
    )
    second = materialize_sector_history_release(
        _observations(),
        contract_path=contract_path,
        output_root=tmp_path / "releases",
        generated_at=generated_at,
    )

    assert first.release_id == second.release_id
    assert first.release_dir == second.release_dir
    assert first.history_path.is_file()
    assert first.rejections_path.is_file()
    assert first.manifest["time_policy"]["retroactive_static_mapping_allowed"] is False
