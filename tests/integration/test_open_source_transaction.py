from __future__ import annotations

from pathlib import Path

import pytest

from alpharank.data.open_source.ingestion import _resolve_open_source_data_layout
from alpharank.data.open_source.storage import OpenSourceLivePaths
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


def test_clean_worktree_symlink_resolves_one_physical_data_root(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "persistent" / "data"
    official = data_root / "open_source" / "official"
    official.mkdir(parents=True)
    worktree = tmp_path / "clean_worktree"
    worktree.mkdir()
    (worktree / "data").symlink_to(data_root, target_is_directory=True)

    resolved_official, open_source_root, resolved_data_root, reference_dir = (
        _resolve_open_source_data_layout(
            project_root=worktree,
            live_dir=worktree / "data" / "open_source" / "official",
            reference_data_dir=worktree / "data",
        )
    )

    assert resolved_official == official.resolve()
    assert open_source_root == (data_root / "open_source").resolve()
    assert resolved_data_root == data_root.resolve()
    assert reference_dir == data_root.resolve()
    paths = OpenSourceLivePaths(
        resolved_official,
        audit_root_dir=open_source_root / "audit",
    )
    assert (paths.audit_dir / "2025").relative_to(paths.root_dir) == Path(
        "audit/2025"
    )
