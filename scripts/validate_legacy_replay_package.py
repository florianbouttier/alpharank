#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_project_root(manifest_path: Path, relative_paths: list[str]) -> Path:
    candidates = list(manifest_path.resolve().parents) + [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if any((candidate / relative_path).exists() for relative_path in relative_paths):
            return candidate
    return Path.cwd()


def validate_manifest(manifest_path: Path, *, strict_code: bool = False) -> tuple[list[str], list[str]]:
    manifest = _load_manifest(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []

    input_snapshot_dir_value = manifest.get("input_snapshot_dir")
    if not input_snapshot_dir_value:
        errors.append("missing input_snapshot_dir")
        input_snapshot_dir = None
    else:
        input_snapshot_dir = Path(input_snapshot_dir_value)
        if not input_snapshot_dir.exists():
            errors.append(f"missing input snapshot directory: {input_snapshot_dir}")

    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        errors.append("missing datasets section")
    else:
        for name, entry in datasets.items():
            if not isinstance(entry, dict):
                errors.append(f"{name}: dataset entry is not an object")
                continue
            path_value = entry.get("canonical_path")
            expected_sha = entry.get("sha256")
            if not path_value:
                errors.append(f"{name}: missing canonical_path")
                continue
            path = Path(path_value)
            if not path.exists():
                errors.append(f"{name}: missing file {path}")
                continue
            if input_snapshot_dir is not None and input_snapshot_dir.exists():
                try:
                    path.relative_to(input_snapshot_dir)
                except ValueError:
                    errors.append(f"{name}: canonical_path is outside input_snapshot_dir: {path}")
            if not expected_sha:
                errors.append(f"{name}: missing sha256")
                continue
            actual_sha = _sha256_path(path)
            if actual_sha != expected_sha:
                errors.append(f"{name}: sha256 mismatch expected={expected_sha} actual={actual_sha}")

    run_config = manifest.get("run_config")
    if not isinstance(run_config, dict):
        errors.append("missing run_config")
    elif not isinstance(run_config.get("source_input_sha256"), dict):
        errors.append("missing run_config.source_input_sha256")

    code_context = manifest.get("code_context")
    if not isinstance(code_context, dict):
        errors.append("missing code_context")
    elif not isinstance(code_context.get("critical_file_sha256"), dict):
        errors.append("missing code_context.critical_file_sha256")
    else:
        project_root = _resolve_project_root(manifest_path, list(code_context["critical_file_sha256"]))
        for relative_path, expected_sha in code_context["critical_file_sha256"].items():
            current_path = project_root / relative_path
            if not current_path.exists():
                warnings.append(f"{relative_path}: current code file is missing")
                continue
            actual_sha = _sha256_path(current_path)
            if actual_sha != expected_sha:
                message = f"{relative_path}: current code sha differs from recorded run sha"
                if strict_code:
                    errors.append(message)
                else:
                    warnings.append(message)

    if manifest.get("open_source_run_id_match") is False:
        errors.append("open_source_run_id_match is false")
    if manifest.get("open_source_output_manifest_run_id_match") is False:
        errors.append("open_source_output_manifest_run_id_match is false")
    if manifest.get("open_source_output_matches_published_snapshot") is False:
        differing_files = manifest.get("open_source_output_published_snapshot_differing_files")
        suffix = f": {differing_files}" if differing_files else ""
        errors.append(f"open_source_output_matches_published_snapshot is false{suffix}")

    return errors, warnings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a retained legacy monthly replay package.")
    parser.add_argument("manifest", help="Path to outputs/YYYY-MM-DD/data_input_manifest.json")
    parser.add_argument(
        "--strict-code",
        action="store_true",
        help="Fail if current critical code hashes differ from the recorded run hashes.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors, warnings = validate_manifest(Path(args.manifest), strict_code=args.strict_code)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Legacy replay package is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
