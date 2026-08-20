"""Versioned pytest suite classification without moving historical tests."""

from __future__ import annotations

import fnmatch
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

TEST_SUITE_POLICY_SCHEMA_VERSION = 1
ALLOWED_TEST_SUITES = ("unit", "integration", "replay", "network", "production")


@dataclass(frozen=True, slots=True)
class SuiteRule:
    """An ordered suite rule; the first matching rule owns the test file."""

    suite: str
    patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SuitePolicy:
    """Validated ordered rules and the default suite."""

    policy_id: str
    default_suite: str
    rules: tuple[SuiteRule, ...]


def load_test_suite_policy(path: Path) -> SuitePolicy:
    """Load the versioned test-suite policy and reject unknown structure."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Test suite policy must be a JSON object")
    expected_keys = {
        "policy_id",
        "schema_version",
        "description",
        "effective_date",
        "default_suite",
        "ordered_rules",
    }
    unknown = set(raw) - expected_keys
    missing = expected_keys - set(raw)
    if unknown or missing:
        raise ValueError(
            f"Test suite policy keys differ: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if raw["schema_version"] != TEST_SUITE_POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported test suite policy schema version")
    policy_id = _require_string(raw["policy_id"], "policy_id")
    default_suite = _require_suite(raw["default_suite"])
    raw_rules = raw["ordered_rules"]
    if not isinstance(raw_rules, list):
        raise ValueError("Test suite ordered_rules must be a list")
    rules = tuple(_parse_rule(rule) for rule in raw_rules)
    configured_suites = {default_suite, *(rule.suite for rule in rules)}
    missing_suites = set(ALLOWED_TEST_SUITES) - configured_suites
    if missing_suites:
        raise ValueError(f"Test suite policy does not classify suites: {sorted(missing_suites)}")
    return SuitePolicy(policy_id=policy_id, default_suite=default_suite, rules=rules)


def classify_test_path(path: str, policy: SuitePolicy) -> str:
    """Return the first matching suite, or the explicit default suite."""

    normalized = Path(path).as_posix()
    for rule in policy.rules:
        if any(fnmatch.fnmatchcase(normalized, pattern) for pattern in rule.patterns):
            return rule.suite
    return policy.default_suite


def build_test_suite_report(paths: Sequence[str], policy: SuitePolicy) -> dict[str, object]:
    """Summarize a deterministic list of files by configured suite."""

    assignments = [
        {"path": path, "suite": classify_test_path(path, policy)} for path in sorted(paths)
    ]
    counts = Counter(str(row["suite"]) for row in assignments)
    return {
        "policy_id": policy.policy_id,
        "file_count": len(assignments),
        "counts": {suite: counts.get(suite, 0) for suite in ALLOWED_TEST_SUITES},
        "assignments": assignments,
    }


def _parse_rule(raw: object) -> SuiteRule:
    if not isinstance(raw, dict) or set(raw) != {"suite", "patterns"}:
        raise ValueError("Each test suite rule must contain only suite and patterns")
    suite = _require_suite(raw["suite"])
    patterns = raw["patterns"]
    if not isinstance(patterns, list) or not patterns:
        raise ValueError(f"Test suite {suite} must declare at least one pattern")
    parsed_patterns = tuple(_require_string(pattern, "pattern") for pattern in patterns)
    return SuiteRule(suite=suite, patterns=parsed_patterns)


def _require_suite(value: object) -> str:
    suite = _require_string(value, "suite")
    if suite not in ALLOWED_TEST_SUITES:
        raise ValueError(f"Unknown test suite: {suite}")
    return suite


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Test suite {label} must be a non-empty string")
    return value
