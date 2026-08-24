"""Differential gate for Python module size, function size, and complexity."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

SCHEMA_VERSION = 1
SCOPE_ROOTS = ("src", "scripts", "tests")
LIBRARY_MODULE_MAXIMUM_LINES = 800
SCRIPT_MODULE_MAXIMUM_LINES = 250
FUNCTION_MAXIMUM_LINES = 80
COMPLEXITY_MAXIMUM = 10
EXCLUDED_DIRECTORIES = frozenset({"_archive", "_old"})
COMPLEXITY_PATTERN = re.compile(
    r"`(?P<symbol>[^`]+)` is too complex \((?P<measured>\d+) > (?P<limit>\d+)\)"
)


@dataclass(frozen=True, slots=True)
class SizeViolation:
    """One measured threshold violation with a stable repository identity."""

    kind: str
    path: str
    symbol: str
    line: int
    measured: int
    limit: int

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.kind, self.path, self.symbol


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.scope: list[str] = []
        self.violations: list[SizeViolation] = []
        self.qualname_by_line: dict[int, str] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join((*self.scope, node.name))
        self.qualname_by_line[node.lineno] = qualname
        start_line = min(
            (decorator.lineno for decorator in node.decorator_list),
            default=node.lineno,
        )
        if node.end_lineno is None:
            raise ValueError(f"Function has no end line: {self.path}:{node.lineno}")
        measured = node.end_lineno - start_line + 1
        if measured > FUNCTION_MAXIMUM_LINES:
            self.violations.append(
                SizeViolation(
                    kind="function_lines",
                    path=self.path,
                    symbol=qualname,
                    line=start_line,
                    measured=measured,
                    limit=FUNCTION_MAXIMUM_LINES,
                )
            )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def build_python_size_baseline(
    root: Path,
    *,
    ruff_executable: str = "ruff",
) -> dict[str, object]:
    """Measure tracked maintained Python and return a deterministic baseline."""

    resolved_root = root.resolve()
    paths = _tracked_python_paths(resolved_root)
    violations, qualnames = _measure_ast_sizes(resolved_root, paths)
    complexity, ruff_version = _measure_complexity(
        resolved_root,
        paths,
        qualnames,
        ruff_executable=ruff_executable,
    )
    rows = sorted((*violations, *complexity), key=lambda row: row.identity)
    counts = Counter(row.kind for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": "python_size_baseline_v1",
        "policy": _policy_payload(),
        "tool_version_at_baseline": ruff_version,
        "summary": {
            "tracked_python_file_count": len(paths),
            "violation_count": len(rows),
            "violations_by_kind": dict(sorted(counts.items())),
        },
        "violations": [asdict(row) for row in rows],
    }


def compare_python_size_baselines(
    baseline: Mapping[str, object],
    current: Mapping[str, object],
) -> dict[str, object]:
    """Reject a new violation or any increase of an existing measurement."""

    expected = _validated_rows(baseline)
    observed = _validated_rows(current)
    policy_changed = baseline.get("policy") != current.get("policy")
    regressions: list[dict[str, object]] = []
    for identity, row in sorted(observed.items()):
        previous = expected.get(identity)
        if previous is None:
            regressions.append({**asdict(row), "reason": "new_violation"})
        elif row.measured > previous.measured:
            regressions.append(
                {
                    **asdict(row),
                    "reason": "measurement_increase",
                    "baseline_measured": previous.measured,
                }
            )
    resolved = [
        asdict(row) for identity, row in sorted(expected.items()) if identity not in observed
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": not policy_changed and not regressions,
        "policy_changed": policy_changed,
        "baseline_violation_count": len(expected),
        "current_violation_count": len(observed),
        "regression_count": len(regressions),
        "resolved_count": len(resolved),
        "regressions": regressions,
        "resolved": resolved,
    }


def load_python_size_baseline(path: Path) -> dict[str, object]:
    """Load and validate one versioned baseline."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Python size baseline must be an object: {path}")
    _validated_rows(raw)
    return raw


def write_python_size_baseline(path: Path, payload: Mapping[str, object]) -> None:
    """Write a deterministic baseline for review."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _measure_ast_sizes(
    root: Path,
    paths: Sequence[str],
) -> tuple[list[SizeViolation], dict[tuple[str, int], str]]:
    violations: list[SizeViolation] = []
    qualnames: dict[tuple[str, int], str] = {}
    for relative_path in paths:
        source = (root / relative_path).read_text(encoding="utf-8")
        line_count = len(source.splitlines())
        module_limit = _module_limit(relative_path)
        if module_limit is not None and line_count > module_limit:
            violations.append(
                SizeViolation(
                    kind="module_lines",
                    path=relative_path,
                    symbol="<module>",
                    line=1,
                    measured=line_count,
                    limit=module_limit,
                )
            )
        collector = _FunctionCollector(relative_path)
        collector.visit(ast.parse(source, filename=relative_path))
        violations.extend(collector.violations)
        qualnames.update(
            {(relative_path, line): name for line, name in collector.qualname_by_line.items()}
        )
    return violations, qualnames


def _measure_complexity(
    root: Path,
    paths: Sequence[str],
    qualnames: Mapping[tuple[str, int], str],
    *,
    ruff_executable: str,
) -> tuple[list[SizeViolation], str]:
    result = subprocess.run(
        [
            ruff_executable,
            "check",
            *SCOPE_ROOTS,
            "--select",
            "C901",
            "--config",
            f"lint.mccabe.max-complexity={COMPLEXITY_MAXIMUM}",
            "--output-format=json",
            "--exit-zero",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    diagnostics = json.loads(result.stdout)
    if not isinstance(diagnostics, list):
        raise ValueError("Ruff complexity output must be a list")
    tracked = set(paths)
    violations = []
    for diagnostic in diagnostics:
        violation = _complexity_violation(root, diagnostic, tracked, qualnames)
        if violation is not None:
            violations.append(violation)
    version = subprocess.run(
        [ruff_executable, "--version"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return violations, version


def _complexity_violation(
    root: Path,
    diagnostic: object,
    tracked: set[str],
    qualnames: Mapping[tuple[str, int], str],
) -> SizeViolation | None:
    if not isinstance(diagnostic, dict):
        raise ValueError("Ruff complexity diagnostic must be an object")
    filename = diagnostic.get("filename")
    message = diagnostic.get("message")
    location = diagnostic.get("location")
    if (
        not isinstance(filename, str)
        or not isinstance(message, str)
        or not isinstance(location, dict)
    ):
        raise ValueError("Ruff complexity diagnostic is incomplete")
    line = location.get("row")
    if not isinstance(line, int):
        raise ValueError("Ruff complexity diagnostic line must be an integer")
    path = Path(filename).resolve().relative_to(root).as_posix()
    if path not in tracked:
        return None
    match = COMPLEXITY_PATTERN.fullmatch(message)
    if match is None:
        raise ValueError(f"Unsupported Ruff complexity message: {message}")
    return SizeViolation(
        kind="complexity",
        path=path,
        symbol=qualnames.get((path, line), match.group("symbol")),
        line=line,
        measured=int(match.group("measured")),
        limit=int(match.group("limit")),
    )


def _tracked_python_paths(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", *SCOPE_ROOTS],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = completed.stdout.decode("utf-8").split("\0")
    return sorted(
        path
        for path in paths
        if path.endswith(".py")
        and (root / path).is_file()
        and not EXCLUDED_DIRECTORIES.intersection(Path(path).parts)
    )


def _module_limit(path: str) -> int | None:
    if path.startswith("src/"):
        return LIBRARY_MODULE_MAXIMUM_LINES
    if path.startswith("scripts/"):
        return SCRIPT_MODULE_MAXIMUM_LINES
    return None


def _policy_payload() -> dict[str, object]:
    return {
        "scope_roots": list(SCOPE_ROOTS),
        "excluded_directories": sorted(EXCLUDED_DIRECTORIES),
        "library_module_maximum_lines": LIBRARY_MODULE_MAXIMUM_LINES,
        "script_module_maximum_lines": SCRIPT_MODULE_MAXIMUM_LINES,
        "function_maximum_lines": FUNCTION_MAXIMUM_LINES,
        "complexity_maximum": COMPLEXITY_MAXIMUM,
        "exception_policy": "owner_approval_required",
    }


def _validated_rows(payload: Mapping[str, object]) -> dict[tuple[str, str, str], SizeViolation]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Python size baseline schema version")
    raw_rows = payload.get("violations")
    if not isinstance(raw_rows, list):
        raise ValueError("Python size baseline violations must be a list")
    rows: dict[tuple[str, str, str], SizeViolation] = {}
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("Python size baseline violation must be an object")
        row = SizeViolation(
            kind=_required_string(raw, "kind"),
            path=_required_string(raw, "path"),
            symbol=_required_string(raw, "symbol"),
            line=_required_integer(raw, "line"),
            measured=_required_integer(raw, "measured"),
            limit=_required_integer(raw, "limit"),
        )
        if row.measured <= row.limit:
            raise ValueError(f"Non-violating Python size row: {row.identity}")
        if row.identity in rows:
            raise ValueError(f"Duplicate Python size violation: {row.identity}")
        rows[row.identity] = row
    return rows


def _required_string(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Python size violation {key} must be a string")
    return value


def _required_integer(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Python size violation {key} must be a positive integer")
    return value
