from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from alpharank.data.publishing.snapshot_storage import copy_snapshot_file


@dataclass(frozen=True)
class _ProtectedPath:
    live: Path
    backup: Path
    existed: bool


class OpenSourceStoreTransaction:
    """Rollback official mutable layers when an ingestion does not complete."""

    def __init__(self, *, official_dir: Path) -> None:
        self.official_dir = official_dir.resolve()
        self.root_dir = self.official_dir.parent
        transaction_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.transaction_dir = self.root_dir / "_transactions" / transaction_id
        self._protected: list[_ProtectedPath] = []
        self._history_children: set[str] = set()

    def __enter__(self) -> OpenSourceStoreTransaction:
        recover_interrupted_transactions(self.root_dir)
        self.transaction_dir.mkdir(parents=True, exist_ok=False)
        candidates = (
            self.official_dir / "raw",
            self.official_dir / "target",
            self.official_dir / "manifests" / "latest_run.json",
            self.official_dir / "manifests" / "raw_store_quarantine.json",
            self.root_dir / "output",
        )
        for index, live in enumerate(candidates):
            backup = self.transaction_dir / "backup" / str(index)
            existed = live.exists()
            if existed:
                _copy_path(live, backup)
            self._protected.append(_ProtectedPath(live=live, backup=backup, existed=existed))

        history_root = self.root_dir / "history" / "output"
        self._history_children = {path.name for path in history_root.iterdir()} if history_root.exists() else set()
        self._write_manifest("running")
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self._write_manifest("committed")
            shutil.rmtree(self.transaction_dir)
            return False
        self.rollback()
        return False

    def rollback(self) -> None:
        for item in self._protected:
            _remove_path(item.live)
            if item.existed:
                _copy_path(item.backup, item.live)
        history_root = self.root_dir / "history" / "output"
        if history_root.exists():
            for path in history_root.iterdir():
                if path.name not in self._history_children:
                    _remove_path(path)
        self._write_manifest("rolled_back")
        shutil.rmtree(self.transaction_dir)

    def _write_manifest(self, status: str) -> None:
        payload = {
            "status": status,
            "official_dir": str(self.official_dir),
            "history_children_before": sorted(self._history_children),
            "protected_paths": [
                {
                    "live": str(item.live),
                    "backup": str(item.backup),
                    "existed": item.existed,
                }
                for item in self._protected
            ],
        }
        (self.transaction_dir / "transaction_manifest.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def recover_interrupted_transactions(root_dir: Path) -> None:
    transactions_root = root_dir / "_transactions"
    if not transactions_root.exists():
        return
    for transaction_dir in sorted(path for path in transactions_root.iterdir() if path.is_dir()):
        manifest_path = transaction_dir / "transaction_manifest.json"
        if not manifest_path.exists():
            shutil.rmtree(transaction_dir)
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "running":
            shutil.rmtree(transaction_dir)
            continue
        for item in payload.get("protected_paths", []):
            live = Path(item["live"])
            backup = Path(item["backup"])
            _remove_path(live)
            if item.get("existed"):
                _copy_path(backup, live)
        history_root = root_dir / "history" / "output"
        history_before = set(payload.get("history_children_before", []))
        if history_root.exists():
            for path in history_root.iterdir():
                if path.name not in history_before:
                    _remove_path(path)
        shutil.rmtree(transaction_dir)


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        def copy_file(src: str, dst: str) -> str:
            copy_snapshot_file(src, dst)
            return dst

        shutil.copytree(source, destination, copy_function=copy_file)
    else:
        copy_snapshot_file(source, destination)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
