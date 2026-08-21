from __future__ import annotations

import ast
from pathlib import Path

from alpharank.quality.code_inventory import (
    _has_main_guard,
    _is_archived_script,
    _resolve_imports,
    validate_code_inventory,
)

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "docs/architecture/code_dependency_inventory_v1.json"


def test_versioned_code_inventory_matches_tracked_graph() -> None:
    report = validate_code_inventory(ROOT, INVENTORY)

    assert report["passed"] is True
    assert report["summary"]["script_count"] >= 100
    assert report["summary"]["library_count"] >= 100
    assert report["summary"]["active_entrypoint_count"] >= 50


def test_imports_and_main_guard_are_resolved_without_importing_code() -> None:
    tree = ast.parse(
        "from alpharank.portfolio import engine\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(0)\n"
    )
    modules = {
        "alpharank.portfolio": "src/alpharank/portfolio/__init__.py",
        "alpharank.portfolio.engine": "src/alpharank/portfolio/engine.py",
    }

    assert _resolve_imports(tree, modules) == {"src/alpharank/portfolio/engine.py"}
    assert _has_main_guard(tree) is True


def test_archive_paths_are_not_active_entrypoints() -> None:
    assert _is_archived_script("scripts/_archive/2026-08-20/old.py") is True
    assert _is_archived_script("scripts/_old/compatibility.py") is True
    assert _is_archived_script("scripts/run_legacy.py") is False
