from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from alpharank.governance import (
    BASELINE_MANIFEST_NAME,
    BaselineValidationError,
    seal_baseline_package,
    validate_baseline_package,
)


def _seal(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "inputs.json").write_text('{"snapshot":"s1"}\n', encoding="utf-8")
    (source / "holdings.csv").write_text("month,ticker,weight\n2020-01,A,1\n", encoding="utf-8")
    package = tmp_path / "v1-audited-biased"
    seal_baseline_package(
        package_dir=package,
        baseline_id="v1-audited-biased",
        sources={"audited_run": source},
        approved_by="owner",
        implementation_commit="a" * 40,
        source_snapshot_id="s1",
        known_limitations=("historical result is not causal proof",),
        sealed_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
    )
    return package


def test_baseline_package_is_immutable(tmp_path: Path) -> None:
    package = _seal(tmp_path)
    report = validate_baseline_package(package)

    assert report["baseline_id"] == "v1-audited-biased"
    assert report["payload_file_count"] == 2
    manifest = json.loads(
        (package / BASELINE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["causal_validation"] is False
    assert len(manifest["inventory"]) == manifest["payload_file_count"]
    assert all(len(row["sha256"]) == 64 for row in manifest["inventory"])
    assert not (package.stat().st_mode & 0o222)
    assert not any(path.stat().st_mode & 0o222 for path in package.rglob("*"))

    sealed_file = package / "payload" / "audited_run" / "holdings.csv"
    with pytest.raises(PermissionError):
        sealed_file.write_text("rewritten\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        seal_baseline_package(
            package_dir=package,
            baseline_id="v1-audited-biased",
            sources={"audited_run": tmp_path / "source"},
            approved_by="owner",
            implementation_commit="b" * 40,
        )


def test_baseline_validator_detects_payload_tampering(tmp_path: Path) -> None:
    package = _seal(tmp_path)
    sealed_file = package / "payload" / "audited_run" / "holdings.csv"
    sealed_file.chmod(0o644)
    sealed_file.write_text("month,ticker,weight\n2020-01,B,1\n", encoding="utf-8")

    with pytest.raises(BaselineValidationError, match="SHA-256 mismatch"):
        validate_baseline_package(package)


def test_baseline_validator_rejects_uninventoried_package_file(
    tmp_path: Path,
) -> None:
    package = _seal(tmp_path)
    package.chmod(0o755)
    unexpected = package / "untracked.txt"
    unexpected.write_text("not inventoried\n", encoding="utf-8")
    unexpected.chmod(0o444)
    package.chmod(0o555)

    with pytest.raises(BaselineValidationError, match="unexpected sealed package file"):
        validate_baseline_package(package)


def test_baseline_is_independent_from_later_source_changes(tmp_path: Path) -> None:
    package = _seal(tmp_path)
    source_file = tmp_path / "source" / "holdings.csv"
    source_file.write_text("changed after sealing\n", encoding="utf-8")

    report = validate_baseline_package(package)
    assert report["passed"] is True
