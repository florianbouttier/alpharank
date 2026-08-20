from __future__ import annotations

from pathlib import Path

from scripts.open_source.compact_output_history import compact_history


def test_compaction_replaces_only_byte_identical_snapshot_files(tmp_path: Path) -> None:
    history = tmp_path / "history" / "output"
    first = history / "snapshot_1"
    second = history / "snapshot_2"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "same.parquet").write_bytes(b"same payload" * 100)
    (second / "same.parquet").write_bytes(b"same payload" * 100)
    (first / "different.parquet").write_bytes(b"first")
    (second / "different.parquet").write_bytes(b"second")

    report = compact_history(history, workers=1)

    assert report["duplicate_group_count"] == 1
    assert report["duplicate_file_count"] == 1
    assert report["replaced_file_count"] == 1
    assert (first / "same.parquet").read_bytes() == (second / "same.parquet").read_bytes()
    assert (first / "different.parquet").read_bytes() != (second / "different.parquet").read_bytes()
