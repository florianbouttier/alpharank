from __future__ import annotations

from pathlib import Path

import pytest

from alpharank.quality.test_suites import classify_test_path, load_test_suite_policy

ROOT = Path(__file__).resolve().parents[1]
POLICY = load_test_suite_policy(ROOT / "configs/quality/test_suites_v1.json")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        relative_path = Path(str(item.path)).resolve().relative_to(ROOT).as_posix()
        suite = classify_test_path(relative_path, POLICY)
        item.add_marker(getattr(pytest.mark, suite))
