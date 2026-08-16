from __future__ import annotations

from pathlib import Path

import pytest

from alpharank.data.open_source.transaction import OpenSourceStoreTransaction


def _seed_store(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "open_source"
    official = root / "official"
    (official / "raw").mkdir(parents=True)
    (official / "target").mkdir(parents=True)
    (official / "manifests").mkdir(parents=True)
    (root / "output").mkdir(parents=True)
    (root / "history" / "output" / "snapshot_old").mkdir(parents=True)
    (official / "raw" / "data.parquet").write_bytes(b"raw before")
    (official / "target" / "data.parquet").write_bytes(b"target before")
    (official / "manifests" / "latest_run.json").write_text('{"run_id":"before"}', encoding="utf-8")
    (root / "output" / "data.parquet").write_bytes(b"output before")
    return root, official


def test_failed_transaction_restores_store_and_removes_new_snapshot(tmp_path: Path) -> None:
    root, official = _seed_store(tmp_path)

    with pytest.raises(RuntimeError, match="fail"):
        with OpenSourceStoreTransaction(official_dir=official):
            (official / "raw" / "data.parquet").write_bytes(b"raw after")
            (official / "target" / "new.parquet").write_bytes(b"new")
            (root / "output" / "data.parquet").write_bytes(b"output after")
            (root / "history" / "output" / "snapshot_new").mkdir()
            raise RuntimeError("fail")

    assert (official / "raw" / "data.parquet").read_bytes() == b"raw before"
    assert not (official / "target" / "new.parquet").exists()
    assert (root / "output" / "data.parquet").read_bytes() == b"output before"
    assert (root / "history" / "output" / "snapshot_old").exists()
    assert not (root / "history" / "output" / "snapshot_new").exists()


def test_successful_transaction_keeps_changes(tmp_path: Path) -> None:
    root, official = _seed_store(tmp_path)

    with OpenSourceStoreTransaction(official_dir=official):
        (official / "raw" / "data.parquet").write_bytes(b"raw after")

    assert (official / "raw" / "data.parquet").read_bytes() == b"raw after"
    assert not (root / "_transactions").exists() or not any((root / "_transactions").iterdir())
