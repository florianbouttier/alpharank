"""Static policy for explicit errors and structured library logging."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Iterable


def tracked_maintained_python_paths(root: Path) -> tuple[Path, ...]:
    """Return tracked Python files governed by the maintained-code policy."""

    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    relative_paths = completed.stdout.decode("utf-8").split("\0")
    return tuple(
        root / relative_path
        for relative_path in relative_paths
        if relative_path
        and (relative_path.startswith("src/") or relative_path.startswith("scripts/"))
        and not relative_path.startswith("scripts/_old/")
    )


def audit_error_handling(
    root: Path,
    paths: Iterable[Path] | None = None,
) -> dict[str, object]:
    """Reject library prints, bare handlers and unaudited broad catches."""

    violations: list[dict[str, object]] = []
    selected_paths = tuple(paths) if paths is not None else tracked_maintained_python_paths(root)
    for path in selected_paths:
        relative_path = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative_path)
        lines = source.splitlines()

        for node in ast.walk(tree):
            if (
                relative_path.startswith("src/alpharank/")
                and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                violations.append(_violation(relative_path, node.lineno, "library_print"))
            if not isinstance(node, ast.ExceptHandler):
                continue
            if node.type is None:
                violations.append(_violation(relative_path, node.lineno, "bare_except"))
                continue
            if not isinstance(node.type, ast.Name) or node.type.id != "Exception":
                continue

            source_line = lines[node.lineno - 1]
            is_process_boundary = (
                relative_path.startswith("scripts/")
                and "process boundary" in source_line
                and _calls_logger_exception(node)
                and _returns_explicit_status(node)
            )
            if not is_process_boundary:
                violations.append(_violation(relative_path, node.lineno, "broad_exception"))

    return {
        "schema_version": 1,
        "passed": not violations,
        "checked_file_count": len(selected_paths),
        "violations": violations,
    }


def _calls_logger_exception(handler: ast.ExceptHandler) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "exception"
        for node in ast.walk(handler)
    )


def _returns_explicit_status(handler: ast.ExceptHandler) -> bool:
    return any(
        isinstance(node, ast.Return) and node.value is not None
        for node in ast.walk(handler)
    )


def _violation(path: str, line: int, code: str) -> dict[str, object]:
    return {"path": path, "line": line, "code": code}
