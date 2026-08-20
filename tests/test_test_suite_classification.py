from __future__ import annotations

from pathlib import Path

from alpharank.quality.test_suites import (
    ALLOWED_TEST_SUITES,
    SuitePolicy,
    SuiteRule,
    build_test_suite_report,
    classify_test_path,
    load_test_suite_policy,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ordered_suite_rules_take_precedence_over_broad_patterns() -> None:
    policy = SuitePolicy(
        policy_id="fixture",
        default_suite="unit",
        rules=(
            SuiteRule("production", ("tests/test_open_source_nightly.py",)),
            SuiteRule("integration", ("tests/test_open_source_*.py",)),
            SuiteRule("network", ("tests/test_network.py",)),
            SuiteRule("replay", ("tests/test_replay.py",)),
        ),
    )

    assert classify_test_path("tests/test_open_source_nightly.py", policy) == "production"
    assert classify_test_path("tests/test_open_source_storage.py", policy) == "integration"
    assert classify_test_path("tests/test_small_function.py", policy) == "unit"


def test_repository_policy_classifies_every_suite_and_test_file() -> None:
    policy = load_test_suite_policy(ROOT / "configs/quality/test_suites_v1.json")
    paths = [path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").glob("test_*.py")]

    report = build_test_suite_report(paths, policy)

    assert report["file_count"] == len(paths)
    assert set(report["counts"]) == set(ALLOWED_TEST_SUITES)
    assert all(count > 0 for count in report["counts"].values())
