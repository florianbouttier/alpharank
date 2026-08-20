from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from alpharank.observability import reset_run_log_context, set_run_log_context
from alpharank.quality.test_suites import classify_test_path, load_test_suite_policy

ROOT = Path(__file__).resolve().parents[1]
POLICY = load_test_suite_policy(ROOT / "configs/quality/test_suites_v1.json")


@pytest.fixture(autouse=True)
def isolate_run_log_context() -> Iterator[None]:
    """Prevent process-global run identifiers from leaking between tests."""

    token = set_run_log_context(
        run_id="not_applicable",
        snapshot_id="not_applicable",
        component="alpharank",
        step="unspecified",
    )
    try:
        yield
    finally:
        reset_run_log_context(token)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        relative_path = Path(str(item.path)).resolve().relative_to(ROOT).as_posix()
        suite = classify_test_path(relative_path, POLICY)
        item.add_marker(getattr(pytest.mark, suite))
