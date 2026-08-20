"""Deterministic inventory of executable scripts and internal Python readers."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

CODE_INVENTORY_SCHEMA_VERSION = 1
PUBLIC_ENTRYPOINTS = (
    "scripts/run_legacy.py",
    "scripts/run_backtest.py",
    "scripts/build_common_legacy_boosting_replay.py",
    "scripts/validate_legacy_replay_package.py",
    "scripts/validate_common_portfolio_engine.py",
    "scripts/validate_documentation.py",
)


def build_code_inventory(root: Path) -> dict[str, object]:
    """Build the repository's tracked script-to-library dependency graph."""

    python_paths = tracked_python_paths(root)
    modules = {_module_name(path): path for path in python_paths}
    basename_paths = _unique_script_basenames(python_paths)
    parsed: dict[str, ast.Module] = {}
    parse_errors: list[str] = []
    for path in python_paths:
        try:
            parsed[path] = ast.parse((root / path).read_text(encoding="utf-8"), filename=path)
        except (OSError, SyntaxError, UnicodeError) as error:
            parse_errors.append(f"{path}: {error}")

    edges: set[tuple[str, str, str]] = set()
    node_rows: list[dict[str, object]] = []
    for path in python_paths:
        tree = parsed.get(path)
        if tree is None:
            continue
        imports = sorted(_resolve_imports(tree, modules))
        commands = sorted(_resolve_script_commands(tree, basename_paths, set(python_paths)))
        for target in imports:
            edges.add((path, target, "import"))
        for target in commands:
            if target != path:
                edges.add((path, target, "command"))
        is_script = path.startswith("scripts/")
        archived = _is_archived_script(path)
        node_rows.append(
            {
                "path": path,
                "module": _module_name(path),
                "layer": "script" if is_script else "library",
                "lifecycle": "archived" if archived else "active",
                "has_main_guard": _has_main_guard(tree) if is_script else False,
                "declared_public_entrypoint": path in PUBLIC_ENTRYPOINTS,
                "imports": imports,
                "invokes": commands,
            }
        )

    readers: dict[str, set[str]] = {path: set() for path in python_paths}
    for source, target, _kind in edges:
        readers[target].add(source)
    for row in node_rows:
        row["readers"] = sorted(readers[str(row["path"])])

    missing_entrypoints = sorted(set(PUBLIC_ENTRYPOINTS) - set(python_paths))
    active_entrypoints = sorted(
        str(row["path"])
        for row in node_rows
        if row["layer"] == "script"
        and row["lifecycle"] == "active"
        and row["has_main_guard"] is True
    )
    edge_rows = [
        {"source": source, "target": target, "kind": kind}
        for source, target, kind in sorted(edges)
    ]
    return {
        "schema_version": CODE_INVENTORY_SCHEMA_VERSION,
        "inventory_id": "alpharank_code_dependency_inventory_v1",
        "scope": ["scripts/**/*.py", "src/alpharank/**/*.py"],
        "source_policy": "tracked_files_only",
        "public_entrypoints": list(PUBLIC_ENTRYPOINTS),
        "active_entrypoints": active_entrypoints,
        "missing_public_entrypoints": missing_entrypoints,
        "parse_errors": parse_errors,
        "summary": {
            "python_file_count": len(python_paths),
            "script_count": sum(path.startswith("scripts/") for path in python_paths),
            "library_count": sum(path.startswith("src/alpharank/") for path in python_paths),
            "active_entrypoint_count": len(active_entrypoints),
            "internal_edge_count": len(edge_rows),
            "unread_node_count": sum(not readers[path] for path in python_paths),
        },
        "nodes": node_rows,
        "edges": edge_rows,
    }


def validate_code_inventory(root: Path, inventory_path: Path) -> dict[str, object]:
    """Compare the versioned inventory with the current tracked Python graph."""

    observed = json.loads(inventory_path.read_text(encoding="utf-8"))
    expected = build_code_inventory(root)
    errors = []
    if observed != expected:
        errors.append("code dependency inventory differs; regenerate it explicitly")
    if expected["parse_errors"]:
        errors.append("one or more tracked Python files could not be parsed")
    if expected["missing_public_entrypoints"]:
        errors.append("one or more declared public entrypoints are missing")
    return {
        "inventory_id": expected["inventory_id"],
        "schema_version": expected["schema_version"],
        "passed": not errors,
        "errors": errors,
        "summary": expected["summary"],
    }


def write_code_inventory(path: Path, inventory: Mapping[str, object]) -> None:
    """Write a stable, reviewable JSON representation of the code graph."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def tracked_python_paths(root: Path) -> list[str]:
    """Return tracked Python paths in the maintained code and script roots."""

    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            "scripts/*.py",
            "scripts/**/*.py",
            "src/alpharank/*.py",
            "src/alpharank/**/*.py",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = completed.stdout.decode("utf-8").split("\0")
    return sorted(path for path in paths if path and (root / path).is_file())


def _resolve_imports(tree: ast.Module, modules: Mapping[str, str]) -> set[str]:
    targets = set()
    for node in ast.walk(tree):
        names: Iterable[str]
        if isinstance(node, ast.Import):
            names = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = (
                f"{node.module}.{alias.name}" if alias.name != "*" else node.module
                for alias in node.names
            )
        else:
            continue
        for imported_name in names:
            target = _longest_module_match(imported_name, modules)
            if target is not None:
                targets.add(target)
    return targets


def _longest_module_match(imported_name: str, modules: Mapping[str, str]) -> str | None:
    parts = imported_name.split(".")
    for length in range(len(parts), 0, -1):
        candidate = ".".join(parts[:length])
        if candidate in modules:
            return modules[candidate]
    return None


def _resolve_script_commands(
    tree: ast.Module,
    basename_paths: Mapping[str, str],
    known_paths: set[str],
) -> set[str]:
    targets = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        raw = node.value.replace("\\", "/")
        if not raw.endswith(".py"):
            continue
        normalized = raw.removeprefix("./")
        if normalized in known_paths:
            targets.add(normalized)
            continue
        name = PurePosixPath(normalized).name
        if name in basename_paths:
            targets.add(basename_paths[name])
    return targets


def _unique_script_basenames(paths: Sequence[str]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for path in paths:
        if path.startswith("scripts/"):
            grouped.setdefault(PurePosixPath(path).name, []).append(path)
    return {name: matches[0] for name, matches in grouped.items() if len(matches) == 1}


def _module_name(path: str) -> str:
    module_path = path[:-3] if path.endswith(".py") else path
    if module_path.startswith("src/"):
        module_path = module_path[4:]
    if module_path.endswith("/__init__"):
        module_path = module_path[: -len("/__init__")]
    return module_path.replace("/", ".")


def _is_archived_script(path: str) -> bool:
    return path.startswith(("scripts/_archive/", "scripts/_old/"))


def _has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        comparison = node.test
        if not isinstance(comparison, ast.Compare) or len(comparison.ops) != 1:
            continue
        if not isinstance(comparison.ops[0], ast.Eq) or len(comparison.comparators) != 1:
            continue
        left = comparison.left
        right = comparison.comparators[0]
        if (
            isinstance(left, ast.Name)
            and left.id == "__name__"
            and isinstance(right, ast.Constant)
            and right.value == "__main__"
        ):
            return True
    return False
