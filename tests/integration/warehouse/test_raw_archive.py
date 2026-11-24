from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from alpharank.data.ingestion import raw_archive
from alpharank.data.ingestion.raw_archive import (
    RawDownloadAttempt,
    archive_raw_frame_delta,
    reconstruct_raw_frame,
    record_raw_download,
    record_raw_download_batch,
    register_immutable_raw_file,
)
from alpharank.data.ingestion.raw_archive_frames import reconstruct_archive_index


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


def test_raw_archive_preserves_v1_hashes_without_python_row_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbid_python_row_maps(**kwargs):
        raise AssertionError("Do not reconstruct millions of RAW rows into Python dictionaries")

    monkeypatch.setattr(raw_archive, "_reconstruct_internal", forbid_python_row_maps, raising=False)
    initial = _prices([("AAPL.US", "2026-08-18", 230.0), ("MSFT.US", "2026-08-18", 510.0)])
    changed = _prices([("AAPL.US", "2026-08-18", 231.0), ("NVDA.US", "2026-08-18", 180.0)])
    first = _archive(tmp_path, "run01", initial)
    second = _archive(tmp_path, "run02", changed)
    third = _archive(tmp_path, "run03", _prices([]))
    restored = _archive(tmp_path, "run04", initial)
    assert (
        first.snapshot_sha256 == "c250a73c550b3bcdf61c23065ee9c814c80e9ee598ebc290bf300ce89b9eec55"
    )
    assert (
        second.snapshot_sha256 == "95ec746688328a2f621bda60be0ef58d2c68b298eeada7a43a9138ae04e27053"
    )
    assert (
        third.snapshot_sha256 == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert restored.snapshot_sha256 == first.snapshot_sha256
    assert restored.restored_row_count == 2
    assert restored.stored_content_row_count == 0
    archive = tmp_path / "raw" / "yahoo" / "prices"
    assert reconstruct_raw_frame(archive_dir=archive, run_id="run02").equals(changed)
    assert reconstruct_raw_frame(archive_dir=archive).equals(initial)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("events", "event hash mismatch"),
        ("snapshot", "snapshot hash mismatch"),
        ("reference", "content reference is unresolved"),
        ("cycle", "parent cycle"),
    ],
)
def test_raw_archive_still_rejects_corruption(
    tmp_path: Path, corruption: str, message: str
) -> None:
    result = _archive(tmp_path, "run01", _prices([("AAPL.US", "2026-08-18", 230.0)]))
    manifest = json.loads(result.manifest_path.read_text())
    path = result.run_dir / "events.parquet"
    if corruption == "snapshot":
        manifest["snapshot_sha256"] = "0" * 64
    elif corruption == "cycle":
        manifest["parent_run_id"] = "run01"
    else:
        events = pl.read_parquet(path)
        if corruption == "events":
            events = events.with_columns(pl.lit(999.0).alias("adjusted_close"))
        else:
            events = events.with_columns(pl.lit(False).alias("stores_content"))
        events.write_parquet(path)
        if corruption == "reference":
            manifest["events_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match=message):
        reconstruct_raw_frame(archive_dir=result.run_dir.parent.parent, run_id="run01")


def test_raw_archive_preserves_typed_values_and_empty_first_observation(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "ticker": ["É.US", "NULL.US", "NAN.US"],
            "date": [date(2026, 8, 18)] * 3,
            "adjusted_close": [float("inf"), None, float("nan")],
            "accepted": [datetime(2026, 8, 18, 12, 30)] * 3,
            "valid": [True, False, None],
            "count": [1, 2, 3],
        }
    ).sort(["ticker", "date"])
    first = _archive(tmp_path, "run01", frame.head(0))
    assert first.input_row_count == first.stored_content_row_count == 0
    _archive(tmp_path, "run02", frame)
    archive = tmp_path / "raw" / "yahoo" / "prices"
    assert reconstruct_raw_frame(archive_dir=archive).equals(frame)
    assert reconstruct_raw_frame(archive_dir=archive, run_id="run01").equals(
        frame.head(0).select(sorted(frame.columns))
    )


def test_raw_archive_rejects_changed_contract_before_writing(tmp_path: Path) -> None:
    _archive(tmp_path, "run01", _prices([("AAPL.US", "2026-08-18", 230.0)]))
    with pytest.raises(ValueError, match="Raw archive contract changed"):
        _archive(tmp_path, "run02", _prices([]).with_columns(pl.lit(1).alias("extra")))
    assert not (tmp_path / "raw" / "yahoo" / "prices" / "runs" / "run02").exists()


def test_non_missing_event_with_null_hash_is_not_silently_a_deletion(tmp_path: Path) -> None:
    result = _archive(tmp_path, "run01", _prices([("AAPL.US", "2026-08-18", 230.0)]))
    path = result.run_dir / "events.parquet"
    pl.read_parquet(path).with_columns(
        pl.lit(None, dtype=pl.String).alias("row_sha256"),
        pl.lit(False).alias("stores_content"),
    ).write_parquet(path)
    manifest = json.loads(result.manifest_path.read_text())
    manifest["events_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["snapshot_sha256"] = hashlib.sha256(b"").hexdigest()
    result.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="content reference is unresolved"):
        reconstruct_raw_frame(archive_dir=result.run_dir.parent.parent, run_id="run01")


def test_missing_event_is_a_deletion_even_with_ignored_legacy_metadata(tmp_path: Path) -> None:
    first = _archive(tmp_path, "run01", _prices([("AAPL.US", "2026-08-18", 230.0)]))
    result = _archive(tmp_path, "run02", _prices([]))
    original_hash = pl.read_parquet(first.run_dir / "events.parquet")["row_sha256"][0]
    path = result.run_dir / "events.parquet"
    pl.read_parquet(path).with_columns(
        pl.lit(original_hash).alias("row_sha256"),
        pl.lit(True).alias("stores_content"),
    ).write_parquet(path)
    manifest = json.loads(result.manifest_path.read_text())
    manifest["events_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result.manifest_path.write_text(json.dumps(manifest))
    restored = reconstruct_raw_frame(archive_dir=result.run_dir.parent.parent, run_id="run02")
    assert restored.is_empty()


def test_raw_archive_does_not_retain_obsolete_string_view_buffers(tmp_path: Path) -> None:
    initial = _prices(
        [(f"T{index:04}.US", "2026-08-18", float(index + 1)) for index in range(1000)]
    ).with_columns(pl.Series("payload", [f"{index:04}:" + "x" * 128 for index in range(1000)]))
    _archive(tmp_path, "run01", initial)
    result = _archive(tmp_path, "run02", initial.head(1))
    archive = result.run_dir.parent.parent
    state, _, _ = reconstruct_archive_index(archive_dir=archive, run_id="run02")
    assert state.height == 1
    hash_buffers = state["row_sha256"].to_arrow(compat_level=pl.CompatLevel.newest()).buffers()
    assert sum(buffer.size for buffer in hash_buffers[2:] if buffer is not None) == 64
    restored = reconstruct_raw_frame(archive_dir=archive, run_id="run02")
    assert restored.equals(initial.head(1))
    payload_buffers = restored["payload"].to_arrow(compat_level=pl.CompatLevel.newest()).buffers()
    assert sum(buffer.size for buffer in payload_buffers[2:] if buffer is not None) == 133


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


def _record_stockanalysis_attempt(
    tmp_path: Path,
    *,
    receipt_id: str,
    retrieved_at: str,
    payload: bytes | None,
    response_status: int,
):
    return record_raw_download(
        archive_dir=tmp_path / "raw" / "stockanalysis",
        receipt_id=receipt_id,
        source_name="stockanalysis",
        dataset_name="daily_price_history",
        request_id=f"stockanalysis:{receipt_id}",
        retrieved_at=retrieved_at,
        response_status=response_status,
        payload=payload,
        payload_format="json",
        requested_scope={"ticker": "AAPL", "range": "Max"},
        ingester_version="test-suite",
        error=None if payload is not None else "service unavailable",
    )


def test_raw_download_receipts_reuse_identical_payload_object(tmp_path: Path) -> None:
    payload = b'{"status":200,"data":[]}'
    first = _record_stockanalysis_attempt(
        tmp_path,
        receipt_id="attempt_01",
        retrieved_at="2026-08-20T10:00:00+00:00",
        payload=payload,
        response_status=200,
    )
    second = _record_stockanalysis_attempt(
        tmp_path,
        receipt_id="attempt_02",
        retrieved_at="2026-08-20T10:01:00+00:00",
        payload=payload,
        response_status=200,
    )

    assert first.payload_sha256 == second.payload_sha256
    assert first.payload_object_path == second.payload_object_path
    assert not first.payload_reused
    assert second.payload_reused
    assert len(list((tmp_path / "raw" / "stockanalysis" / "objects").glob("*/*"))) == 1
    manifest = json.loads(second.provider_manifest_path.read_text(encoding="utf-8"))
    assert manifest["receipt_count"] == 2
    assert manifest["payload_object_count"] == 1
    assert manifest["latest_receipt_id"] == "attempt_02"
    assert manifest["validation"] == {
        "payload_objects": "passed",
        "receipt_contract": "passed",
    }


def test_raw_download_batch_rebuilds_manifest_once_for_all_attempts(tmp_path: Path) -> None:
    archive_dir = tmp_path / "raw" / "stockanalysis"
    results = record_raw_download_batch(
        archive_dir=archive_dir,
        source_name="stockanalysis",
        dataset_name="daily_price_history",
        ingester_version="test-suite",
        attempts=(
            RawDownloadAttempt(
                receipt_id="batch_01",
                request_id="stockanalysis:batch_01",
                retrieved_at="2026-08-20T10:00:00+00:00",
                response_status=200,
                payload=b"{}",
                payload_format="json",
                requested_scope={"ticker": "AAPL"},
            ),
            RawDownloadAttempt(
                receipt_id="batch_02",
                request_id="stockanalysis:batch_02",
                retrieved_at="2026-08-20T10:01:00+00:00",
                response_status=200,
                payload=b"{}",
                payload_format="json",
                requested_scope={"ticker": "MSFT"},
            ),
        ),
    )

    manifest = json.loads(results[-1].provider_manifest_path.read_text(encoding="utf-8"))
    assert len(results) == 2
    assert manifest["receipt_count"] == 2
    assert manifest["payload_object_count"] == 1
    assert manifest["latest_receipt_id"] == "batch_02"


def test_failed_raw_download_attempt_keeps_receipt_without_payload(tmp_path: Path) -> None:
    failed = _record_stockanalysis_attempt(
        tmp_path,
        receipt_id="attempt_failed",
        retrieved_at="2026-08-20T10:00:00+00:00",
        payload=None,
        response_status=503,
    )

    receipt = json.loads(failed.receipt_path.read_text(encoding="utf-8"))
    assert receipt["response_status"] == 503
    assert receipt["payload_sha256"] is None
    assert receipt["payload_object_path"] is None
    assert receipt["size_bytes"] == 0
    assert receipt["error"] == "service unavailable"


def test_raw_download_receipt_id_is_immutable(tmp_path: Path) -> None:
    kwargs = {
        "receipt_id": "attempt_01",
        "retrieved_at": "2026-08-20T10:00:00+00:00",
        "payload": b"{}",
        "response_status": 200,
    }
    _record_stockanalysis_attempt(tmp_path, **kwargs)

    with pytest.raises(FileExistsError, match="receipt already exists"):
        _record_stockanalysis_attempt(tmp_path, **kwargs)
