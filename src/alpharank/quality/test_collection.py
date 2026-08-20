"""Path-independent Pytest collection signatures for safe test moves."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def build_test_body_signature(paths: Sequence[Path]) -> dict[str, object]:
    """Hash test-function ASTs independently from their module location."""

    tests: list[dict[str, object]] = []
    assertion_count = 0
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            body = ast.dump(node, annotate_fields=True, include_attributes=False)
            test_assertion_count = sum(
                isinstance(child, ast.Assert) for child in ast.walk(node)
            )
            assertion_count += test_assertion_count
            tests.append(
                {
                    "name": node.name,
                    "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                    "assertion_count": test_assertion_count,
                }
            )

    tests.sort(key=lambda row: str(row["name"]))
    names = [str(row["name"]) for row in tests]
    if len(names) != len(set(names)):
        raise ValueError("Test body signature requires unique test-function names")
    serialized = json.dumps(
        tests,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "test_count": len(tests),
        "assertion_count": assertion_count,
        "test_ast_sha256": hashlib.sha256(serialized).hexdigest(),
    }


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
