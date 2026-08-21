from pathlib import Path

import polars as pl
from scripts.open_source.build_sec_output_package import (
    _overlay_identity_remediation,
    _overlay_legacy_identity_outputs,
)

LINEAGE_FILES = {
    "general_reference.parquet": ["ticker", "name"],
    "general_reference_lineage.parquet": ["ticker", "name"],
    "financials_sec_consolidated.parquet": ["ticker", "metric", "value"],
    "financials_sec_lineage.parquet": ["ticker", "metric", "value"],
    "earnings_sec_consolidated.parquet": ["ticker", "epsActual"],
    "earnings_sec_lineage.parquet": ["ticker", "epsActual"],
    "earnings_sec_long.parquet": ["ticker", "metric", "value"],
}


def test_identity_overlay_freezes_every_non_target_row(tmp_path: Path) -> None:
    previous_lineage = tmp_path / "previous" / "lineage"
    previous_lineage.mkdir(parents=True)
    previous_frames: dict[str, pl.DataFrame] = {}
    candidate_frames: list[pl.DataFrame] = []
    for file_name, columns in LINEAGE_FILES.items():
        previous = _frame(columns, unaffected_value=1.0, target_ticker="SNDK.US")
        candidate = _frame(columns, unaffected_value=999.0, target_ticker="SNDK_OLD.US")
        previous.write_parquet(previous_lineage / file_name)
        previous_frames[file_name] = previous
        candidate_frames.append(candidate)

    registry = pl.DataFrame(
        {
            "source_ticker": ["SNDK", "SNDK"],
            "canonical_ticker": ["SNDK_OLD", "SNDK"],
            "security_id": ["old", "new"],
            "issuer_cik": ["0001000180", "0002023554"],
            "valid_from": ["1990-01-01", "2025-02-24"],
            "valid_to": ["2016-05-12", None],
            "identity_status": ["historical", "current"],
            "evidence": ["old evidence", "new evidence"],
        }
    )
    *published, report = _overlay_identity_remediation(
        previous_output_dir=tmp_path / "previous",
        registry=registry,
        general_reference=candidate_frames[0],
        general_reference_lineage=candidate_frames[1],
        consolidated_financials=candidate_frames[2],
        consolidated_lineage=candidate_frames[3],
        earnings_consolidated=candidate_frames[4],
        earnings_lineage=candidate_frames[5],
        earnings_long=candidate_frames[6],
    )

    for frame in published:
        unaffected = frame.filter(pl.col("ticker") == "AAA.US")
        target = frame.filter(pl.col("ticker") == "SNDK_OLD.US")
        assert unaffected.get_column("value").item() == 1.0
        assert target.get_column("value").item() == 20.0
        assert frame.filter(pl.col("ticker") == "SNDK.US").is_empty()
    assert report["mode"] == "registered_security_identity_only"
    assert report["published_tickers"] == ["SNDK.US", "SNDK_OLD.US"]


def test_legacy_identity_overlay_freezes_published_non_target_rows(tmp_path: Path) -> None:
    previous_dir = tmp_path / "previous"
    candidate_dir = tmp_path / "candidate"
    previous_dir.mkdir()
    candidate_dir.mkdir()
    previous = pl.DataFrame(
        {"Ticker": ["AAA.US", "SNDK.US"], "date": ["2025-03-31"] * 2, "value": [1, 2]}
    )
    candidate = pl.DataFrame(
        {
            "Ticker": ["AAA.US", "SNDK_OLD.US"],
            "date": ["2025-03-31", "2015-03-31"],
            "value": [999, 20],
        }
    )
    previous.write_parquet(previous_dir / "example.parquet")
    candidate_path = candidate_dir / "example.parquet"
    candidate.write_parquet(candidate_path)
    registry = pl.DataFrame(
        {
            "source_ticker": ["SNDK", "SNDK"],
            "canonical_ticker": ["SNDK_OLD", "SNDK"],
            "security_id": ["old", "new"],
            "issuer_cik": ["0001000180", "0002023554"],
            "valid_from": ["1990-01-01", "2025-02-24"],
            "valid_to": ["2016-05-12", None],
            "identity_status": ["historical", "current"],
            "evidence": ["old evidence", "new evidence"],
        }
    )

    report = _overlay_legacy_identity_outputs(
        previous_output_dir=previous_dir,
        candidate_paths={"example.parquet": candidate_path},
        registry=registry,
    )

    published = pl.read_parquet(candidate_path)
    assert published.filter(pl.col("Ticker") == "AAA.US").get_column("value").item() == 1
    assert published.filter(pl.col("Ticker") == "SNDK_OLD.US").get_column("value").item() == 20
    assert report["example.parquet"]["frozen_non_target_rows"] == 1


def _frame(
    requested_columns: list[str],
    *,
    unaffected_value: float,
    target_ticker: str,
) -> pl.DataFrame:
    data: dict[str, list[object]] = {
        "ticker": ["AAA.US", target_ticker],
        "value": [unaffected_value, 20.0],
        "name": [str(unaffected_value), "target"],
        "metric": ["revenue", "revenue"],
        "epsActual": [unaffected_value, 20.0],
    }
    frame = pl.DataFrame({column: data[column] for column in requested_columns})
    if "value" not in frame.columns:
        frame = frame.with_columns(
            pl.when(pl.col("ticker") == "AAA.US")
            .then(pl.lit(unaffected_value))
            .otherwise(pl.lit(20.0))
            .alias("value")
        )
    return frame
