from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from alpharank.data.open_source.raw_archive import (
    archive_raw_frame_delta,
    reconstruct_raw_frame,
    register_immutable_raw_file,
)


def _prices(rows: list[tuple[str, str, float | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"ticker": pl.String, "date": pl.String, "adjusted_close": pl.Float64},
        orient="row",
    )


def _archive(tmp_path: Path, run_id: str, frame: pl.DataFrame):
    return archive_raw_frame_delta(
        archive_dir=tmp_path / "raw" / "yahoo" / "prices",
        run_id=run_id,
        frame=frame,
        key_columns=("ticker", "date"),
        source="yahoo",
        dataset="prices",
        observed_at=f"2026-08-{run_id[-2:]}T02:15:00+00:00",
        request={"start_date": "2005-01-01", "end_date": "2026-08-20"},
    )


def test_identical_raw_download_stores_no_price_rows_twice(tmp_path: Path) -> None:
    frame = _prices(
        [
            ("AAPL.US", "2026-08-18", 230.0),
            ("MSFT.US", "2026-08-18", 510.0),
        ]
    )
    first = _archive(tmp_path, "run01", frame)
    second = _archive(tmp_path, "run02", frame)

    assert first.stored_content_row_count == 2
    assert second.stored_content_row_count == 0
    assert second.unchanged_row_count == 2
    assert pl.read_parquet(second.run_dir / "events.parquet").is_empty()
    assert reconstruct_raw_frame(archive_dir=tmp_path / "raw" / "yahoo" / "prices").equals(frame)


def test_raw_archive_traces_changed_new_and_missing_rows(tmp_path: Path) -> None:
    initial = _prices(
        [
            ("AAPL.US", "2026-08-18", 230.0),
            ("MSFT.US", "2026-08-18", 510.0),
        ]
    )
    changed = _prices(
        [
            ("AAPL.US", "2026-08-18", 231.0),
            ("NVDA.US", "2026-08-18", 180.0),
        ]
    )
    _archive(tmp_path, "run01", initial)
    result = _archive(tmp_path, "run02", changed)
    events = pl.read_parquet(result.run_dir / "events.parquet")

    assert result.updated_row_count == 1
    assert result.inserted_row_count == 1
    assert result.missing_row_count == 1
    assert set(events["event_type"].to_list()) == {"updated", "inserted", "missing"}
    assert reconstruct_raw_frame(archive_dir=tmp_path / "raw" / "yahoo" / "prices").equals(changed)


def test_restored_identical_row_references_old_raw_content(tmp_path: Path) -> None:
    initial = _prices([("AAPL.US", "2026-08-18", 230.0)])
    _archive(tmp_path, "run01", initial)
    _archive(tmp_path, "run02", _prices([]))
    restored = _archive(tmp_path, "run03", initial)
    events = pl.read_parquet(restored.run_dir / "events.parquet")

    assert restored.restored_row_count == 1
    assert restored.stored_content_row_count == 0
    assert events.select("event_type", "stores_content").to_dicts() == [
        {"event_type": "restored", "stores_content": False}
    ]
    assert reconstruct_raw_frame(archive_dir=tmp_path / "raw" / "yahoo" / "prices").equals(initial)


def test_raw_archive_rejects_duplicate_business_keys(tmp_path: Path) -> None:
    duplicate = _prices(
        [
            ("AAPL.US", "2026-08-18", 230.0),
            ("AAPL.US", "2026-08-18", 231.0),
        ]
    )
    with pytest.raises(ValueError, match="duplicate keys"):
        _archive(tmp_path, "run01", duplicate)


def test_immutable_eodhd_content_is_stored_once_for_multiple_source_ids(tmp_path: Path) -> None:
    source_a = tmp_path / "source_a.parquet"
    source_b = tmp_path / "source_b.parquet"
    payload = _prices([("AAPL.US", "2026-08-18", 230.0)])
    payload.write_parquet(source_a)
    source_b.write_bytes(source_a.read_bytes())
    archive_dir = tmp_path / "raw" / "eodhd" / "prices"

    first_manifest = register_immutable_raw_file(
        archive_dir=archive_dir,
        source_id="eodhd_01",
        source_path=source_a,
        source="eodhd",
        dataset="US_Finalprice",
        observed_at="2026-08-18T00:00:00+00:00",
    )
    second_manifest = register_immutable_raw_file(
        archive_dir=archive_dir,
        source_id="eodhd_02",
        source_path=source_b,
        source="eodhd",
        dataset="US_Finalprice",
        observed_at="2026-08-19T00:00:00+00:00",
    )

    first = json.loads(first_manifest.read_text())
    second = json.loads(second_manifest.read_text())
    assert first["sha256"] == second["sha256"]
    assert first["object_path"] == second["object_path"]
    assert len(list((archive_dir / "objects").glob("*/*"))) == 1
