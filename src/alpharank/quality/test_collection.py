"""Path-independent Pytest collection signatures for safe test moves."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def collect_canonical_node_ids(root: Path, test_paths: Sequence[str]) -> list[str]:
    """Collect tests and remove their parent directory from each node id."""

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *test_paths,
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    node_ids = []
    for line in completed.stdout.splitlines():
        candidate = line.strip()
        if "::" not in candidate or candidate.startswith(("=", "<")):
            continue
        path, *selectors = candidate.split("::")
        node_ids.append("::".join((Path(path).name, *selectors)))
    return sorted(node_ids)


def build_collection_registry(node_ids: Sequence[str]) -> dict[str, object]:
    """Build a reviewable exact baseline with a compact integrity hash."""

    canonical = sorted(node_ids)
    payload = "\n".join(canonical).encode("utf-8")
    return {
        "schema_version": 1,
        "registry_id": "alpharank_test_collection_v1",
        "canonicalization": "test filename plus selectors; parent directories ignored",
        "node_count": len(canonical),
        "node_ids_sha256": hashlib.sha256(payload).hexdigest(),
        "node_ids": canonical,
    }


def write_collection_registry(path: Path, registry: dict[str, object]) -> None:
    """Write the exact collection baseline in deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
