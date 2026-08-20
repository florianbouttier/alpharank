"""Inventory explicitly declared Pytest fixtures without importing tests."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path


def discover_pytest_fixtures(root: Path) -> list[dict[str, object]]:
    """Return deterministic metadata for fixtures declared under ``tests``."""

    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", "tests/*.py", "tests/**/*.py"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = sorted(
        path for path in completed.stdout.decode().split("\0") if path.endswith(".py")
    )
    fixtures: list[dict[str, object]] = []
    for relative_path in paths:
        tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            declaration = _fixture_declaration(node)
            if declaration is None:
                continue
            fixtures.append(
                {
                    "path": relative_path,
                    "name": node.name,
                    **declaration,
                }
            )
    return sorted(fixtures, key=lambda row: (str(row["path"]), str(row["name"])))


def _fixture_declaration(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, object] | None:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if not _is_fixture_name(target):
            continue
        autouse = False
        scope = "function"
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg == "autouse" and isinstance(keyword.value, ast.Constant):
                    autouse = keyword.value.value is True
                if keyword.arg == "scope" and isinstance(keyword.value, ast.Constant):
                    if isinstance(keyword.value.value, str):
                        scope = keyword.value.value
        return {"autouse": autouse, "scope": scope}
    return None


def _is_fixture_name(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "fixture"
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "pytest"
        and node.attr == "fixture"
    )
