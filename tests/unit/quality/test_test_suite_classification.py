from __future__ import annotations

import json
from pathlib import Path

from alpharank.quality.test_fixtures import discover_pytest_fixtures
from alpharank.quality.test_suites import (
    ALLOWED_TEST_SUITES,
    SuitePolicy,
    SuiteRule,
    build_test_suite_report,
    classify_test_path,
    load_test_suite_policy,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_INVENTORY = ROOT / "docs" / "architecture" / "test_fixture_inventory_v1.json"


def test_ordered_suite_rules_take_precedence_over_broad_patterns() -> None:
    policy = SuitePolicy(
        policy_id="fixture",
        default_suite="unit",
        rules=(
            SuiteRule("production", ("tests/production/test_*.py",)),
            SuiteRule("network", ("tests/integration/network/test_*.py",)),
            SuiteRule("integration", ("tests/integration/test_*.py",)),
            SuiteRule("replay", ("tests/replay/test_*.py",)),
        ),
    )

    assert classify_test_path("tests/production/test_open_source_nightly.py", policy) == "production"
    assert classify_test_path("tests/integration/network/test_yahoo.py", policy) == "network"
    assert classify_test_path("tests/integration/test_storage.py", policy) == "integration"
    assert classify_test_path("tests/unit/test_small_function.py", policy) == "unit"


def test_repository_policy_classifies_every_suite_and_test_file() -> None:
    policy = load_test_suite_policy(ROOT / "configs/quality/test_suites_v1.json")
    paths = [path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").rglob("test_*.py")]

    report = build_test_suite_report(paths, policy)

    assert report["file_count"] == len(paths)
    assert set(report["counts"]) == set(ALLOWED_TEST_SUITES)
    assert all(count > 0 for count in report["counts"].values())


def test_only_cross_suite_state_is_declared_as_a_shared_fixture() -> None:
    inventory = json.loads(FIXTURE_INVENTORY.read_text(encoding="utf-8"))

    assert discover_pytest_fixtures(ROOT) == inventory["fixtures"]
    assert inventory["policy"]["default_ownership"] == "local_to_test_module"
