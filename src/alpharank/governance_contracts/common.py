"""Filesystem and hashing primitives shared by governance contracts."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


def canonical_json_sha256(value: Any) -> str:
    """Return a deterministic SHA-256 for JSON-compatible content."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_path(path: Path) -> str:
    """Hash one file without loading it entirely in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(root: Path) -> list[Path]:
    """List regular files recursively in deterministic order."""

    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def directory_hashes(root: Path) -> dict[str, str]:
    """Inventory all files below an existing directory."""

    if not root.is_dir():
        raise FileNotFoundError(f"Version directory not found: {root}")
    return {path.relative_to(root).as_posix(): sha256_path(path) for path in files_under(root)}


def atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through an atomic same-directory replacement."""

    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def promotion_timestamp(value: datetime | None) -> str:
    """Normalize an explicit or current promotion timestamp to UTC."""

    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("Promotion timestamps must include a timezone.")
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
