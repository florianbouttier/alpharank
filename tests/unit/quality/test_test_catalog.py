from __future__ import annotations

import json
from pathlib import Path

from alpharank.quality.test_catalog import (
    ALLOWED_TEST_DOMAINS,
    build_test_catalog,
    classify_test_domain,
    tracked_test_paths,
)
from alpharank.quality.test_collection import (
    build_test_body_signature,
    collect_canonical_node_ids,
)
from alpharank.quality.test_suites import classify_test_path, load_test_suite_policy

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "docs" / "architecture" / "test_catalog_v1.json"
POLICY = ROOT / "configs" / "quality" / "test_suites_v1.json"
COLLECTION = ROOT / "docs" / "architecture" / "test_collection_v1.json"
SPLIT_AUDIT = ROOT / "docs" / "architecture" / "test_split_audit_v1.json"


def test_versioned_catalog_covers_every_tracked_test_file() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    policy = load_test_suite_policy(POLICY)
    rows = catalog["files"]
    paths = tracked_test_paths(ROOT)

    assert [row["path"] for row in rows] == paths
    assert catalog["summary"]["file_count"] == len(paths)
    assert all(row["domain"] == classify_test_domain(row["path"]) for row in rows)
    assert all(row["suite"] == classify_test_path(row["path"], policy) for row in rows)
    assert all(row["test_case_count"] > 0 for row in rows)
    assert all(row["duration_seconds"] >= 0.0 for row in rows)
    assert set(catalog["summary"]["domain_counts"]) == set(ALLOWED_TEST_DOMAINS)

    collection = json.loads(COLLECTION.read_text(encoding="utf-8"))
    assert collect_canonical_node_ids(ROOT, paths) == collection["node_ids"]


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


def test_catalog_reads_nested_suite_junit_classnames(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite><testcase classname="tests.integration.test_example" '
        'name="test_example" time="0.125" /></testsuite>',
        encoding="utf-8",
    )
    policy = load_test_suite_policy(POLICY)

    catalog = build_test_catalog(
        ["tests/integration/test_example.py"],
        policy,
        junit_path=junit,
        measured_at="2026-08-20T00:00:00Z",
        measurement_command="pytest fixture",
    )

    assert catalog["files"][0]["test_case_count"] == 1
    assert catalog["files"][0]["duration_seconds"] == 0.125
    assert catalog["files"][0]["observed_outcome"] == "passed"


def test_monolithic_test_splits_preserve_test_bodies_and_assertions() -> None:
    audit = json.loads(SPLIT_AUDIT.read_text(encoding="utf-8"))

    for split in audit["splits"]:
        observed = build_test_body_signature(
            [ROOT / path for path in split["target_paths"]]
        )
        assert observed == split["before_signature"]
