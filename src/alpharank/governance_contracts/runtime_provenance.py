"""Runtime provenance capture and validation contract."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from alpharank.governance_contracts.common import (
    canonical_json_sha256 as _canonical_json_sha256,
)
from alpharank.governance_contracts.common import (
    sha256_path as _sha256_path,
)
from alpharank.governance_contracts.contracts import (
    RUNTIME_PROVENANCE_CONTRACT_VERSION,
    RuntimeProvenanceError,
)


def capture_runtime_provenance(
    *,
    project_root: Path,
    entrypoint: str,
    command_argv: list[str] | tuple[str, ...],
    resolved_config: Mapping[str, Any],
    seeds: Mapping[str, Any],
    critical_files: list[str] | tuple[str, ...],
    data_identifiers: Mapping[str, Any],
    patch_path: Path,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    """Capture a replay-oriented and truthfully dirty runtime description.

    The patch bundle stores the complete tracked Git patch plus SHA-256 and size
    for every untracked file. This keeps dirty research reproducible without
    embedding possibly large or sensitive untracked payloads in the manifest.
    """

    root = Path(project_root).resolve()
    if not (root / ".git").exists():
        raise RuntimeProvenanceError(f"Not a Git worktree: {root}")
    if not str(entrypoint).strip():
        raise ValueError("entrypoint must be non-empty.")
    if not command_argv:
        raise ValueError("command_argv must be non-empty.")
    timestamp = captured_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("captured_at must include an explicit timezone.")

    git = _capture_git_state(root)
    dependencies = _installed_distributions()
    critical_hashes: dict[str, str] = {}
    missing_critical_files: list[str] = []
    for raw_path in sorted(set(critical_files)):
        path = (root / raw_path).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Critical file escapes project root: {raw_path}") from exc
        if path.is_file():
            critical_hashes[relative] = _sha256_path(path)
        else:
            missing_critical_files.append(relative)
    if missing_critical_files:
        raise RuntimeProvenanceError(
            "Missing critical runtime files: " + ", ".join(missing_critical_files)
        )

    bundle = {
        "runtime_provenance_contract_version": RUNTIME_PROVENANCE_CONTRACT_VERSION,
        "git_head": git["head"],
        "git_branch": git["branch"],
        "tracked_diff_sha256": git["tracked_diff_sha256"],
        "tracked_diff_bytes": git["tracked_diff_bytes"],
        "tracked_diff": git.pop("tracked_diff"),
        "untracked_files": git["untracked_files"],
    }
    patch = Path(patch_path).resolve()
    patch.parent.mkdir(parents=True, exist_ok=True)
    temporary_patch = patch.with_name(f".{patch.name}.tmp-{uuid4().hex}")
    temporary_patch.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_patch.replace(patch)

    provenance = {
        "runtime_provenance_contract_version": RUNTIME_PROVENANCE_CONTRACT_VERSION,
        "captured_at_utc": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "entrypoint": str(entrypoint),
        "command": {
            "argv": [str(value) for value in command_argv],
            "shell_escaped": shlex.join(str(value) for value in command_argv),
        },
        "git": {key: value for key, value in git.items() if key != "untracked_files"},
        "patch_artifact": {
            "path": str(patch),
            "sha256": _sha256_path(patch),
            "size_bytes": patch.stat().st_size,
            "contains": "tracked binary patch and untracked file fingerprints",
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "dependencies": dependencies,
        "dependencies_sha256": _canonical_json_sha256(dependencies),
        "resolved_config": _sanitize_runtime_value(dict(resolved_config)),
        "seeds": _sanitize_runtime_value(dict(seeds)),
        "critical_file_sha256": critical_hashes,
        "data_identifiers": _sanitize_runtime_value(dict(data_identifiers)),
    }
    validate_runtime_provenance(provenance, project_root=root)
    return provenance


def validate_runtime_provenance(
    provenance: Mapping[str, Any], *, project_root: Path | None = None
) -> dict[str, Any]:
    """Fail closed on an incomplete or falsely clean runtime declaration."""

    errors: list[str] = []
    required_top_level = {
        "runtime_provenance_contract_version",
        "captured_at_utc",
        "entrypoint",
        "command",
        "git",
        "patch_artifact",
        "runtime",
        "dependencies",
        "dependencies_sha256",
        "resolved_config",
        "seeds",
        "critical_file_sha256",
        "data_identifiers",
    }
    missing = sorted(required_top_level - set(provenance))
    if missing:
        errors.append("missing runtime provenance fields: " + ", ".join(missing))
    if provenance.get("runtime_provenance_contract_version") != RUNTIME_PROVENANCE_CONTRACT_VERSION:
        errors.append("unsupported runtime provenance contract version")

    git = provenance.get("git")
    if not isinstance(git, Mapping):
        errors.append("git runtime provenance is missing")
        git = {}
    required_git = {
        "head",
        "branch",
        "dirty",
        "status_porcelain_sha256",
        "status_entry_count",
        "tracked_diff_sha256",
        "tracked_diff_bytes",
        "untracked_file_count",
        "untracked_inventory_sha256",
    }
    missing_git = sorted(required_git - set(git))
    if missing_git:
        errors.append("missing git provenance fields: " + ", ".join(missing_git))

    for mapping_name in (
        "command",
        "runtime",
        "dependencies",
        "resolved_config",
        "seeds",
        "critical_file_sha256",
        "data_identifiers",
        "patch_artifact",
    ):
        value = provenance.get(mapping_name)
        if not isinstance(value, Mapping) or not value:
            errors.append(f"{mapping_name} must be a non-empty mapping")
    command = provenance.get("command")
    if isinstance(command, Mapping) and not command.get("argv"):
        errors.append("command.argv must be non-empty")
    if isinstance(provenance.get("dependencies"), Mapping):
        if _canonical_json_sha256(provenance["dependencies"]) != provenance.get(
            "dependencies_sha256"
        ):
            errors.append("dependency inventory SHA-256 mismatch")

    patch_artifact = provenance.get("patch_artifact")
    if isinstance(patch_artifact, Mapping):
        patch = Path(str(patch_artifact.get("path", "")))
        if not patch.is_file():
            errors.append("runtime Git patch artifact is missing")
        elif _sha256_path(patch) != patch_artifact.get("sha256"):
            errors.append("runtime Git patch artifact SHA-256 mismatch")

    if project_root is not None:
        current = _capture_git_state(Path(project_root).resolve())
        if bool(git.get("dirty")) != bool(current["dirty"]):
            errors.append("git_dirty declaration does not match the current worktree")
        for field in (
            "head",
            "status_porcelain_sha256",
            "tracked_diff_sha256",
            "untracked_inventory_sha256",
        ):
            if git.get(field) != current[field]:
                errors.append(f"Git runtime provenance changed: {field}")

    if errors:
        raise RuntimeProvenanceError("; ".join(errors))
    return {
        "passed": True,
        "git_head": git["head"],
        "git_dirty": git["dirty"],
        "dependencies_sha256": provenance["dependencies_sha256"],
        "patch_sha256": provenance["patch_artifact"]["sha256"],
    }


def _git_command(root: Path, *args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        check=False,
        text=not binary,
        timeout=30,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace") if binary else result.stderr
        raise RuntimeProvenanceError(
            f"Git command failed ({' '.join(args)}): {str(stderr).strip()}"
        )
    return result.stdout


def _capture_git_state(root: Path) -> dict[str, Any]:
    status = str(_git_command(root, "status", "--porcelain=v1", "--untracked-files=all"))
    tracked_diff = bytes(
        _git_command(
            root,
            "diff",
            "--binary",
            "--full-index",
            "HEAD",
            "--",
            binary=True,
        )
    )
    untracked_output = str(_git_command(root, "ls-files", "--others", "--exclude-standard"))
    untracked_files: list[dict[str, Any]] = []
    for relative in sorted(line for line in untracked_output.splitlines() if line):
        path = root / relative
        if path.is_file():
            untracked_files.append(
                {
                    "relative_path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_path(path),
                }
            )
    untracked_inventory_sha256 = _canonical_json_sha256(untracked_files)
    return {
        "head": str(_git_command(root, "rev-parse", "HEAD")).strip(),
        "branch": str(_git_command(root, "rev-parse", "--abbrev-ref", "HEAD")).strip(),
        "dirty": bool(status),
        "status_porcelain_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "status_entry_count": len(status.splitlines()),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "tracked_diff_bytes": len(tracked_diff),
        "tracked_diff": tracked_diff.decode("utf-8", errors="surrogateescape"),
        "untracked_file_count": len(untracked_files),
        "untracked_inventory_sha256": untracked_inventory_sha256,
        "untracked_files": untracked_files,
    }


def _installed_distributions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[str(name).lower()] = distribution.version
    return dict(sorted(packages.items()))


def _sanitize_runtime_value(value: Any, *, key: str = "") -> Any:
    sensitive_tokens = (
        "api_key",
        "apikey",
        "password",
        "passwd",
        "secret",
        "token",
        "credential",
        "private_key",
    )
    normalized_key = key.lower().replace("-", "_")
    if any(token in normalized_key for token in sensitive_tokens):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(child_key): _sanitize_runtime_value(child_value, key=str(child_key))
            for child_key, child_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_runtime_value(item, key=key) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
