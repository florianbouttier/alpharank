"""Policy and deterministic inventory for Python files stored per directory."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

POLICY_SCHEMA_VERSION = 1
INVENTORY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DirectoryException:
    """Owner-approved limit override for exactly one repository directory."""

    directory: str
    maximum_files: int
    approved_by: str
    approved_on: str
    approval_reference: str
    reason: str


@dataclass(frozen=True, slots=True)
class PythonDirectoryPolicy:
    """Maximum direct Python-file count and its explicitly approved overrides."""

    policy_id: str
    scope_roots: tuple[str, ...]
    counted_filename_pattern: str
    maximum_files_per_directory: int
    effective_after_task: str
    approved_exceptions: tuple[DirectoryException, ...]


def load_python_directory_policy(path: Path) -> PythonDirectoryPolicy:
    """Load the strict versioned policy or reject an unverifiable exception."""

    raw = _require_mapping(json.loads(path.read_text(encoding="utf-8")), "policy")
    expected_keys = {
        "approved_exceptions",
        "counted_filename_pattern",
        "effective_after_task",
        "maximum_files_per_directory",
        "policy_id",
        "schema_version",
        "scope_roots",
    }
    _require_exact_keys(raw, expected_keys, "policy")
    if raw["schema_version"] != POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported Python directory policy schema_version")
    maximum = _require_positive_integer(
        raw["maximum_files_per_directory"], "maximum_files_per_directory"
    )
    pattern = _require_string(raw["counted_filename_pattern"], "counted_filename_pattern")
    if pattern != "*.py":
        raise ValueError("counted_filename_pattern must remain '*.py'")
    scope_roots = _require_string_sequence(raw["scope_roots"], "scope_roots")
    exceptions = _load_exceptions(raw["approved_exceptions"], maximum, scope_roots)
    return PythonDirectoryPolicy(
        policy_id=_require_string(raw["policy_id"], "policy_id"),
        scope_roots=scope_roots,
        counted_filename_pattern=pattern,
        maximum_files_per_directory=maximum,
        effective_after_task=_require_string(raw["effective_after_task"], "effective_after_task"),
        approved_exceptions=exceptions,
    )


def build_python_directory_inventory(
    root: Path,
    policy: PythonDirectoryPolicy,
) -> dict[str, object]:
    """Count tracked Python files directly stored in every maintained directory."""

    tracked_paths = _tracked_python_paths(root, policy.scope_roots)
    counts: dict[str, int] = {}
    for path in tracked_paths:
        directory = Path(path).parent.as_posix()
        counts[directory] = counts.get(directory, 0) + 1
    exception_by_directory = {item.directory: item for item in policy.approved_exceptions}
    directories = []
    violations = []
    for directory, count in sorted(counts.items()):
        exception = exception_by_directory.get(directory)
        limit = exception.maximum_files if exception else policy.maximum_files_per_directory
        row = {
            "directory": directory,
            "python_file_count": count,
            "limit": limit,
            "exception_reference": exception.approval_reference if exception else None,
            "passed": count <= limit,
        }
        directories.append(row)
        if count > limit:
            violations.append(row)
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "inventory_id": "alpharank_python_directory_inventory_v1",
        "source_policy": "tracked_files_only",
        "policy_id": policy.policy_id,
        "maximum_files_per_directory": policy.maximum_files_per_directory,
        "approved_exception_count": len(policy.approved_exceptions),
        "directory_count": len(directories),
        "python_file_count": len(tracked_paths),
        "violation_count": len(violations),
        "violations": violations,
        "directories": directories,
    }


def validate_python_directory_inventory(
    root: Path,
    policy_path: Path,
    inventory_path: Path,
    *,
    enforce_limit: bool,
) -> dict[str, object]:
    """Compare the versioned inventory and optionally reject every over-limit directory."""

    policy = load_python_directory_policy(policy_path)
    expected = build_python_directory_inventory(root, policy)
    observed = json.loads(inventory_path.read_text(encoding="utf-8"))
    raw_violations = expected["violations"]
    if not isinstance(raw_violations, list):
        raise ValueError("Generated Python directory violations must be a list")
    violations = [
        _require_mapping(item, f"violations[{index}]")
        for index, item in enumerate(raw_violations)
    ]
    errors = []
    if observed != expected:
        errors.append("Python directory inventory differs; regenerate it explicitly")
    if enforce_limit and violations:
        paths = ", ".join(str(row["directory"]) for row in violations)
        errors.append(f"Python directory limit exceeded: {paths}")
    return {
        "inventory_id": expected["inventory_id"],
        "schema_version": expected["schema_version"],
        "passed": not errors,
        "enforce_limit": enforce_limit,
        "errors": errors,
        "python_file_count": expected["python_file_count"],
        "directory_count": expected["directory_count"],
        "violation_count": expected["violation_count"],
        "violations": violations,
    }


def write_python_directory_inventory(path: Path, inventory: Mapping[str, object]) -> None:
    """Write the deterministic directory inventory for review and CI validation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _tracked_python_paths(root: Path, scope_roots: Sequence[str]) -> list[str]:
    command = ["git", "ls-files", "-z", "--"]
    for scope_root in scope_roots:
        command.extend((f"{scope_root}/*.py", f"{scope_root}/**/*.py"))
    completed = subprocess.run(command, cwd=root, check=True, capture_output=True)
    paths = completed.stdout.decode("utf-8").split("\0")
    return sorted(path for path in paths if path and (root / path).is_file())


def _load_exceptions(
    raw: object,
    default_maximum: int,
    scope_roots: Sequence[str],
) -> tuple[DirectoryException, ...]:
    if not isinstance(raw, list):
        raise ValueError("approved_exceptions must be a list")
    expected_keys = {
        "approval_reference",
        "approved_by",
        "approved_on",
        "directory",
        "maximum_files",
        "reason",
    }
    exceptions = []
    seen_directories = set()
    for index, item in enumerate(raw):
        mapping = _require_mapping(item, f"approved_exceptions[{index}]")
        _require_exact_keys(mapping, expected_keys, f"approved_exceptions[{index}]")
        directory = _require_string(mapping["directory"], "directory").rstrip("/")
        if directory in seen_directories:
            raise ValueError(f"Duplicate approved exception for {directory}")
        if not any(directory == scope or directory.startswith(f"{scope}/") for scope in scope_roots):
            raise ValueError(f"Exception directory is outside policy scope: {directory}")
        maximum_files = _require_positive_integer(mapping["maximum_files"], "maximum_files")
        if maximum_files <= default_maximum:
            raise ValueError(f"Exception for {directory} must raise the default maximum")
        exceptions.append(
            DirectoryException(
                directory=directory,
                maximum_files=maximum_files,
                approved_by=_require_string(mapping["approved_by"], "approved_by"),
                approved_on=_require_string(mapping["approved_on"], "approved_on"),
                approval_reference=_require_string(
                    mapping["approval_reference"], "approval_reference"
                ),
                reason=_require_string(mapping["reason"], "reason"),
            )
        )
        seen_directories.add(directory)
    return tuple(exceptions)


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(f"{label} keys differ: missing={missing}, unknown={unknown}")


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object")
    return value


def _require_positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_string_sequence(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    strings = tuple(_require_string(item, label).rstrip("/") for item in value)
    if len(strings) != len(set(strings)):
        raise ValueError(f"{label} contains duplicates")
    return strings
