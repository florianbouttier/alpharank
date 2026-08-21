from __future__ import annotations

from pathlib import Path

import pytest

from alpharank.data.lineage.legacy_archive_policy import (
    assert_legacy_path_not_writable,
    build_legacy_archive_policy,
    validate_legacy_archive_policy,
)


def _catalog(root: Path) -> dict[str, object]:
    return {
        "contract": "alpharank_historical_root_catalog_summary_v1",
        "catalog_id": "catalog-01",
        "file_count": 2,
        "total_source_bytes": 12,
        "roots": [
            {
                "root_id": "legacy_prices",
                "source_path": str(root / "data" / "prices.parquet"),
                "source_kind": "file",
                "file_count": 1,
                "size_bytes": 12,
                "inventory_sha256": "abc123",
            }
        ],
    }


def _readers() -> dict[str, object]:
    return {
        "contract": "alpharank_data_reader_migration_v1",
        "composition_id": "mart-01",
    }


def test_archive_policy_defines_observation_and_exact_rollback(tmp_path: Path) -> None:
    policy = build_legacy_archive_policy(
        tmp_path,
        _catalog(tmp_path),
        _readers(),
        observation_started_at="2026-08-20",
        minimum_observation_days=30,
    )

    report = validate_legacy_archive_policy(
        tmp_path,
        _catalog(tmp_path),
        _readers(),
        policy,
    )

    assert report["archive_not_before"] == "2026-09-19"
    assert report["payload_moved"] is False
    assert report["payload_deleted"] is False
    assert policy["rollback"]["available"] is True


def test_frozen_legacy_root_rejects_governed_write(tmp_path: Path) -> None:
    policy = build_legacy_archive_policy(
        tmp_path,
        _catalog(tmp_path),
        _readers(),
        observation_started_at="2026-08-20",
    )

    with pytest.raises(PermissionError, match="frozen legacy root"):
        assert_legacy_path_not_writable(
            tmp_path,
            tmp_path / "data" / "prices.parquet",
            policy,
        )

    assert_legacy_path_not_writable(
        tmp_path,
        tmp_path / "data" / "warehouse" / "mart" / "prices.parquet",
        policy,
    )
