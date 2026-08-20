from __future__ import annotations

from pathlib import Path

from alpharank.governance_contracts.run_retention import (
    build_run_retention_report,
    validate_run_retention_report,
)


def test_retention_report_measures_exact_duplicates_without_deletion(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    first = outputs / "legacy" / "run_1" / "result.bin"
    second = outputs / "legacy" / "run_2" / "result.bin"
    unique = outputs / "legacy" / "run_3" / "result.bin"
    for path in (first, second, unique):
        path.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"duplicate")
    second.write_bytes(b"duplicate")
    unique.write_bytes(b"different")

    report = build_run_retention_report(outputs, generated_at="2026-08-20")
    validation = validate_run_retention_report(outputs, report)

    assert validation["duplicate_group_count"] == 1
    assert validation["duplicate_file_count"] == 1
    assert validation["reclaimable_bytes"] == len(b"duplicate")
    assert validation["deletion_count"] == 0
    assert report["retention_proposal"]["automatic_deletion"] is False
    assert first.is_file() and second.is_file() and unique.is_file()


def test_same_size_different_bytes_are_not_duplicates(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    first = outputs / "family" / "run_1" / "one.bin"
    second = outputs / "family" / "run_2" / "two.bin"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"abc")
    second.write_bytes(b"xyz")

    report = build_run_retention_report(outputs, generated_at="2026-08-20")

    assert report["size_collision_file_count"] == 2
    assert report["duplicate_group_count"] == 0
    assert report["reclaimable_bytes"] == 0
