from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from alpharank.quality.python_directories import (
    build_python_directory_inventory,
    load_python_directory_policy,
)


def _write_policy(path: Path, *, exceptions: list[dict[str, object]] | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "approved_exceptions": exceptions or [],
                "counted_filename_pattern": "*.py",
                "effective_after_task": "CODEORG-005",
                "maximum_files_per_directory": 2,
                "policy_id": "fixture_policy_v1",
                "schema_version": 1,
                "scope_roots": ["src"],
            }
        ),
        encoding="utf-8",
    )


def _tracked_fixture(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    package = tmp_path / "src" / "package"
    package.mkdir(parents=True)
    for name in ("one.py", "two.py", "three.py"):
        (package / name).write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "src"], cwd=tmp_path, check=True)


def test_inventory_rejects_a_directory_above_the_default_limit(tmp_path: Path) -> None:
    _tracked_fixture(tmp_path)
    policy_path = tmp_path / "policy.json"
    _write_policy(policy_path)

    inventory = build_python_directory_inventory(
        tmp_path,
        load_python_directory_policy(policy_path),
    )

    assert inventory["violation_count"] == 1
    assert inventory["violations"][0]["directory"] == "src/package"
    assert inventory["violations"][0]["python_file_count"] == 3


def test_exception_requires_explicit_owner_approval_fields(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    _write_policy(
        policy_path,
        exceptions=[
            {
                "directory": "src/package",
                "maximum_files": 3,
                "reason": "Temporary compatibility modules",
            }
        ],
    )

    with pytest.raises(ValueError, match="keys differ"):
        load_python_directory_policy(policy_path)
