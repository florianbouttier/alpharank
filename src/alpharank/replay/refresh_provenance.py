"""Precise code, configuration, and runtime provenance comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_provenance_pairs(
    pairs: dict[str, tuple[Path, Path]],
) -> dict[str, Any]:
    """Compare replay manifest provenance without treating output paths as config."""

    compared = {name: _provenance_pair(*paths) for name, paths in pairs.items()}
    return {
        "stages": compared,
        "all_code_identical": all(item["code_identical"] for item in compared.values()),
        "all_config_identical": all(item["config_identical"] for item in compared.values()),
        "all_runtime_identical": all(item["runtime_identical"] for item in compared.values()),
    }


def stable_config(value: Any, key: str = "") -> Any:
    """Remove run locations while retaining economically relevant parameters."""

    ignored = {
        "captured_at_utc",
        "data_dir",
        "input_snapshot_storage",
        "output_dir",
        "run_dir",
        "run_instance_id",
        "run_output_dir",
        "source_input_sha256",
    }
    if key in ignored or key.endswith("_path") or key.endswith("_files") or key.endswith("_dir"):
        return None
    if isinstance(value, dict):
        return {
            name: stable
            for name, nested in sorted(value.items())
            if (stable := stable_config(nested, name)) is not None
        }
    if isinstance(value, list):
        return [stable_config(item) for item in value]
    return value


def mapping_differences(baseline: Any, candidate: Any) -> list[dict[str, Any]]:
    """Return exact JSON paths and before/after values for two objects."""

    baseline_flat = _flatten(baseline)
    candidate_flat = _flatten(candidate)
    differences = []
    for path in sorted(set(baseline_flat) | set(candidate_flat)):
        baseline_value = baseline_flat.get(path, _MISSING)
        candidate_value = candidate_flat.get(path, _MISSING)
        if baseline_value == candidate_value:
            continue
        differences.append(
            {
                "path": path,
                "baseline": _present(baseline_value),
                "candidate": _present(candidate_value),
            }
        )
    return differences


def _provenance_pair(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = _runtime_provenance(baseline_path)
    candidate = _runtime_provenance(candidate_path)
    git_differences = mapping_differences(
        _git_code_context(baseline.get("git", {})),
        _git_code_context(candidate.get("git", {})),
    )
    critical_file_differences = mapping_differences(
        baseline.get("critical_file_sha256", {}),
        candidate.get("critical_file_sha256", {}),
    )
    config_differences = mapping_differences(
        stable_config(baseline.get("resolved_config", {})),
        stable_config(candidate.get("resolved_config", {})),
    )
    runtime_differences = mapping_differences(
        _runtime_context(baseline),
        _runtime_context(candidate),
    )
    return {
        "baseline_manifest": str(baseline_path.resolve()),
        "candidate_manifest": str(candidate_path.resolve()),
        "code_identical": not git_differences and not critical_file_differences,
        "config_identical": not config_differences,
        "runtime_identical": not runtime_differences,
        "git_differences": git_differences,
        "critical_file_differences": critical_file_differences,
        "config_differences": config_differences,
        "runtime_differences": runtime_differences,
    }


def _runtime_provenance(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing replay manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("runtime_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Manifest has no runtime_provenance object: {path}")
    return provenance


def _git_code_context(git: dict[str, Any]) -> dict[str, Any]:
    return {
        key: git.get(key)
        for key in (
            "head",
            "tracked_diff_sha256",
            "tracked_diff_bytes",
            "untracked_inventory_sha256",
        )
    }


def _runtime_context(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "runtime": provenance.get("runtime", {}),
        "dependencies": provenance.get("dependencies", {}),
        "dependencies_sha256": provenance.get("dependencies_sha256"),
        "seeds": provenance.get("seeds", {}),
    }


def _flatten(value: Any, prefix: str = "$") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened = {}
        for name, nested in sorted(value.items()):
            flattened.update(_flatten(nested, f"{prefix}.{name}"))
        return flattened
    return {prefix: value}


def _present(value: Any) -> Any:
    return "<missing>" if value is _MISSING else value


_MISSING = object()
