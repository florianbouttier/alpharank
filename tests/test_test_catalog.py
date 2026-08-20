from __future__ import annotations

import json
from pathlib import Path

from alpharank.quality.test_catalog import (
    ALLOWED_TEST_DOMAINS,
    classify_test_domain,
    tracked_test_paths,
)
from alpharank.quality.test_suites import classify_test_path, load_test_suite_policy

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs" / "architecture" / "test_catalog_v1.json"
POLICY = ROOT / "configs" / "quality" / "test_suites_v1.json"


def test_versioned_catalog_covers_every_tracked_test_file() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    policy = load_test_suite_policy(POLICY)
    rows = catalog["files"]
    paths = tracked_test_paths(ROOT)

    assert [row["path"] for row in rows] == paths
    assert catalog["summary"]["file_count"] == len(paths)
    assert all(row["domain"] == classify_test_domain(row["path"]) for row in rows)
    assert all(row["suite"] == classify_test_path(row["path"], policy) for row in rows)
    assert all(
        row["test_case_count"] > 0 or row["path"] == "tests/test_test_catalog.py"
        for row in rows
    )
    assert all(row["duration_seconds"] >= 0.0 for row in rows)
    assert set(catalog["summary"]["domain_counts"]) == set(ALLOWED_TEST_DOMAINS)


def test_network_and_failed_measurements_are_explicit() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    rows = catalog["files"]

    assert all(
        row["network_requirement"] == "provider_boundary_disabled_by_default"
        for row in rows
        if row["suite"] == "network"
    )
    assert catalog["summary"]["failure_count"] == 3
    assert catalog["summary"]["outcome_counts"]["failed_missing_local_artifacts"] == 2
