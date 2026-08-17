"""Seal and verify immutable methodology baselines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import Any, Mapping
from uuid import uuid4

import polars as pl

from alpharank.data.snapshot_storage import copy_snapshot_file


BASELINE_MANIFEST_NAME = "baseline_manifest.json"
BASELINE_SEAL_NAME = "baseline_manifest.sha256"
BASELINE_CONTRACT_VERSION = 1
ECONOMIC_PREFIX_CONTRACT_VERSION = 1
RUNTIME_PROVENANCE_CONTRACT_VERSION = 1
APPROVED_NUMERIC_TOLERANCE = 1e-12

HOLDINGS_PREFIX_KEYS = (
    "strategy",
    "decision_month",
    "holding_month",
    "ticker",
)
HOLDINGS_PREFIX_NUMERIC_COLUMNS = (
    "target_weight",
    "realized_return",
    "benchmark_return",
)
HOLDINGS_PREFIX_EXACT_COLUMNS = (
    "selection_rank",
    "sector",
    "return_resolution",
    "terminal_event_id",
)
MONTHLY_PREFIX_KEYS = ("strategy", "decision_month", "holding_month")
MONTHLY_PREFIX_NUMERIC_COLUMNS = (
    "gross_return",
    "turnover",
    "transaction_cost",
    "net_return",
    "benchmark_return",
    "active_return",
    "relative_return",
)
MONTHLY_PREFIX_EXACT_COLUMNS = ("n_positions", "sector_count")


class BaselineValidationError(RuntimeError):
    """Raised when a sealed methodology baseline is incomplete or modified."""


class EconomicPrefixError(RuntimeError):
    """Raised when a supposedly neutral migration changes published economics."""


class RuntimeProvenanceError(RuntimeError):
    """Raised when a run manifest does not prove its runtime provenance."""


@dataclass(frozen=True)
class _InventoryEntry:
    relative_path: str
    size_bytes: int
    sha256: str
    storage_mode: str


def seal_baseline_package(
    *,
    package_dir: Path,
    baseline_id: str,
    sources: Mapping[str, Path],
    approved_by: str,
    implementation_commit: str,
    methodology_status: str = "audited_biased_not_causal",
    source_snapshot_id: str | None = None,
    known_limitations: tuple[str, ...] = (),
    sealed_at: datetime | None = None,
) -> dict[str, Any]:
    """Copy audited artifacts into a new write-once baseline package.

    The destination must not exist. Every source file is copied to an
    independent path, preferring APFS copy-on-write clones, and every payload
    file is inventoried. The completed directory is atomically renamed and all
    write bits are removed only after the manifest and its detached seal exist.
    """

    destination = package_dir.resolve()
    if destination.exists():
        raise FileExistsError(
            f"Baseline package already exists and cannot be overwritten: {destination}"
        )
    identifier = str(baseline_id).strip()
    if not identifier:
        raise ValueError("baseline_id must be non-empty.")
    if not sources:
        raise ValueError("At least one baseline source is required.")
    normalized_sources = _validate_sources(sources)
    timestamp = sealed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("sealed_at must include an explicit timezone.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    inventory: list[_InventoryEntry] = []
    try:
        payload_dir = temporary / "payload"
        payload_dir.mkdir()
        for label, source in normalized_sources.items():
            target_root = payload_dir / label
            if source.is_dir():
                target_root.mkdir()
                for source_file in _files_under(source):
                    relative = source_file.relative_to(source)
                    target = target_root / relative
                    storage_mode = copy_snapshot_file(source_file, target)
                    inventory.append(
                        _inventory_entry(
                            target,
                            package_root=temporary,
                            storage_mode=storage_mode,
                        )
                    )
            else:
                target_root.mkdir()
                target = target_root / source.name
                storage_mode = copy_snapshot_file(source, target)
                inventory.append(
                    _inventory_entry(
                        target,
                        package_root=temporary,
                        storage_mode=storage_mode,
                    )
                )

        inventory.sort(key=lambda entry: entry.relative_path)
        root_sha256 = _inventory_sha256(inventory)
        manifest = {
            "baseline_contract_version": BASELINE_CONTRACT_VERSION,
            "baseline_id": identifier,
            "methodology_status": methodology_status,
            "causal_validation": False,
            "sealed_at_utc": timestamp.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "approved_by": str(approved_by),
            "implementation_commit": str(implementation_commit),
            "source_snapshot_id": source_snapshot_id,
            "known_limitations": list(known_limitations),
            "storage_contract": {
                "strategy": "copy_on_write_with_physical_copy_fallback",
                "semantics": "independent byte-identical immutable payload paths",
                "storage_mode_counts": _storage_mode_counts(inventory),
            },
            "source_roots": {
                label: str(path) for label, path in normalized_sources.items()
            },
            "payload_file_count": len(inventory),
            "payload_size_bytes": sum(entry.size_bytes for entry in inventory),
            "payload_inventory_sha256": root_sha256,
            "inventory": [
                {
                    "relative_path": entry.relative_path,
                    "size_bytes": entry.size_bytes,
                    "sha256": entry.sha256,
                    "storage_mode": entry.storage_mode,
                }
                for entry in inventory
            ],
        }
        manifest_path = temporary / BASELINE_MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_sha256 = _sha256_path(manifest_path)
        (temporary / BASELINE_SEAL_NAME).write_text(
            manifest_sha256 + "  " + BASELINE_MANIFEST_NAME + "\n",
            encoding="utf-8",
        )
        _remove_write_bits(temporary)
        temporary.rename(destination)
    except Exception:
        _make_tree_owner_writable(temporary)
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    validate_baseline_package(destination)
    return manifest


def validate_baseline_package(package_dir: Path) -> dict[str, Any]:
    """Fail closed if a sealed baseline differs from its detached inventory."""

    root = package_dir.resolve()
    manifest_path = root / BASELINE_MANIFEST_NAME
    seal_path = root / BASELINE_SEAL_NAME
    errors: list[str] = []
    if not root.is_dir():
        raise BaselineValidationError(f"Baseline package does not exist: {root}")
    if not manifest_path.is_file():
        errors.append(f"missing {BASELINE_MANIFEST_NAME}")
    if not seal_path.is_file():
        errors.append(f"missing {BASELINE_SEAL_NAME}")
    if errors:
        raise BaselineValidationError("; ".join(errors))

    expected_manifest_sha = seal_path.read_text(encoding="utf-8").split()[0]
    actual_manifest_sha = _sha256_path(manifest_path)
    if expected_manifest_sha != actual_manifest_sha:
        errors.append("baseline manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineValidationError("baseline manifest is not valid JSON") from exc

    if manifest.get("baseline_contract_version") != BASELINE_CONTRACT_VERSION:
        errors.append("unsupported baseline contract version")
    if manifest.get("causal_validation") is not False:
        errors.append("audited biased baseline must not claim causal validation")
    inventory_rows = manifest.get("inventory")
    if not isinstance(inventory_rows, list) or not inventory_rows:
        errors.append("baseline payload inventory is missing or empty")
        inventory_rows = []

    expected_by_path: dict[str, dict[str, Any]] = {}
    for row in inventory_rows:
        if not isinstance(row, dict) or not row.get("relative_path"):
            errors.append("invalid baseline inventory row")
            continue
        relative_path = str(row["relative_path"])
        if relative_path in expected_by_path:
            errors.append(f"duplicate baseline inventory path: {relative_path}")
        expected_by_path[relative_path] = row

    payload_dir = root / "payload"
    actual_paths = {
        path.relative_to(root).as_posix() for path in _files_under(payload_dir)
    }
    all_package_files = {
        path.relative_to(root).as_posix() for path in _files_under(root)
    }
    allowed_package_files = actual_paths | {
        BASELINE_MANIFEST_NAME,
        BASELINE_SEAL_NAME,
    }
    for unexpected in sorted(all_package_files - allowed_package_files):
        errors.append(f"unexpected sealed package file: {unexpected}")
    expected_paths = set(expected_by_path)
    for missing in sorted(expected_paths - actual_paths):
        errors.append(f"missing sealed payload file: {missing}")
    for unexpected in sorted(actual_paths - expected_paths):
        errors.append(f"unexpected sealed payload file: {unexpected}")

    actual_entries: list[_InventoryEntry] = []
    for relative_path in sorted(expected_paths & actual_paths):
        path = root / relative_path
        row = expected_by_path[relative_path]
        actual_sha = _sha256_path(path)
        actual_size = path.stat().st_size
        if actual_sha != row.get("sha256"):
            errors.append(f"sealed payload SHA-256 mismatch: {relative_path}")
        if actual_size != row.get("size_bytes"):
            errors.append(f"sealed payload size mismatch: {relative_path}")
        actual_entries.append(
            _InventoryEntry(
                relative_path=relative_path,
                size_bytes=actual_size,
                sha256=actual_sha,
                storage_mode=str(row.get("storage_mode", "unknown")),
            )
        )
    if len(inventory_rows) != manifest.get("payload_file_count"):
        errors.append("payload_file_count does not match inventory")
    if sum(entry.size_bytes for entry in actual_entries) != manifest.get(
        "payload_size_bytes"
    ):
        errors.append("payload_size_bytes does not match inventory")
    if actual_entries and _inventory_sha256(actual_entries) != manifest.get(
        "payload_inventory_sha256"
    ):
        errors.append("payload inventory SHA-256 mismatch")

    for path in [root, manifest_path, seal_path, payload_dir, *root.rglob("*")]:
        if path.exists() and path.stat().st_mode & 0o222:
            errors.append(f"sealed baseline path remains writable: {path.relative_to(root)}")

    if errors:
        raise BaselineValidationError("; ".join(errors))
    return {
        "baseline_id": manifest["baseline_id"],
        "manifest_sha256": actual_manifest_sha,
        "payload_inventory_sha256": manifest["payload_inventory_sha256"],
        "payload_file_count": manifest["payload_file_count"],
        "payload_size_bytes": manifest["payload_size_bytes"],
        "passed": True,
    }


def compare_economic_prefix(
    *,
    reference_holdings: pl.DataFrame,
    candidate_holdings: pl.DataFrame,
    reference_monthly: pl.DataFrame,
    candidate_monthly: pl.DataFrame,
    through_holding_month: str | None = None,
    numeric_tolerance: float = APPROVED_NUMERIC_TOLERANCE,
    tolerance_justification: str | None = (
        "owner-approved floating serialization tolerance; structural decisions remain exact"
    ),
) -> dict[str, Any]:
    """Compare the already-published economic prefix of two portfolio packages.

    The reference calendar defines the prefix unless an earlier explicit cutoff
    is supplied. Candidate rows after that cutoff are ignored. Keys and
    decision-like fields are exact; approved numeric fields use one documented
    absolute tolerance.
    """

    tolerance = float(numeric_tolerance)
    if tolerance < 0.0:
        raise ValueError("numeric_tolerance must be non-negative.")
    if tolerance > 0.0 and not str(tolerance_justification or "").strip():
        raise ValueError("A positive numeric tolerance requires a justification.")

    reference_holdings_normalized = _normalize_economic_frame(
        reference_holdings,
        keys=HOLDINGS_PREFIX_KEYS,
        numeric_columns=HOLDINGS_PREFIX_NUMERIC_COLUMNS,
        exact_candidates=HOLDINGS_PREFIX_EXACT_COLUMNS,
        label="reference holdings",
    )
    candidate_holdings_normalized = _normalize_economic_frame(
        candidate_holdings,
        keys=HOLDINGS_PREFIX_KEYS,
        numeric_columns=HOLDINGS_PREFIX_NUMERIC_COLUMNS,
        exact_candidates=HOLDINGS_PREFIX_EXACT_COLUMNS,
        label="candidate holdings",
    )
    reference_monthly_normalized = _normalize_economic_frame(
        reference_monthly,
        keys=MONTHLY_PREFIX_KEYS,
        numeric_columns=MONTHLY_PREFIX_NUMERIC_COLUMNS,
        exact_candidates=MONTHLY_PREFIX_EXACT_COLUMNS,
        label="reference monthly returns",
    )
    candidate_monthly_normalized = _normalize_economic_frame(
        candidate_monthly,
        keys=MONTHLY_PREFIX_KEYS,
        numeric_columns=MONTHLY_PREFIX_NUMERIC_COLUMNS,
        exact_candidates=MONTHLY_PREFIX_EXACT_COLUMNS,
        label="candidate monthly returns",
    )
    reference_end = reference_monthly_normalized.get_column("holding_month").max()
    if reference_end is None:
        raise ValueError("Reference monthly returns are empty.")
    cutoff = (
        datetime.fromisoformat(through_holding_month).date()
        if through_holding_month is not None
        else reference_end
    )
    if cutoff > reference_end:
        raise ValueError(
            "through_holding_month cannot extend beyond the reference prefix."
        )

    frames = {
        "holdings": (
            reference_holdings_normalized.filter(pl.col("holding_month") <= cutoff),
            candidate_holdings_normalized.filter(pl.col("holding_month") <= cutoff),
            HOLDINGS_PREFIX_KEYS,
            HOLDINGS_PREFIX_NUMERIC_COLUMNS,
            HOLDINGS_PREFIX_EXACT_COLUMNS,
        ),
        "monthly": (
            reference_monthly_normalized.filter(pl.col("holding_month") <= cutoff),
            candidate_monthly_normalized.filter(pl.col("holding_month") <= cutoff),
            MONTHLY_PREFIX_KEYS,
            MONTHLY_PREFIX_NUMERIC_COLUMNS,
            MONTHLY_PREFIX_EXACT_COLUMNS,
        ),
    }
    frame_reports: dict[str, Any] = {}
    for label, (reference, candidate, keys, numeric, exact) in frames.items():
        frame_reports[label] = _compare_economic_frame(
            reference=reference,
            candidate=candidate,
            keys=keys,
            numeric_columns=numeric,
            exact_candidates=exact,
            tolerance=tolerance,
        )

    passed = all(report["passed"] for report in frame_reports.values())
    return {
        "economic_prefix_contract_version": ECONOMIC_PREFIX_CONTRACT_VERSION,
        "through_holding_month": str(cutoff),
        "numeric_tolerance": tolerance,
        "tolerance_justification": tolerance_justification,
        "structural_comparison": "exact",
        "frames": frame_reports,
        "passed": passed,
    }


def require_stable_economic_prefix(**kwargs: Any) -> dict[str, Any]:
    """Return the comparison report or fail a neutral migration closed."""

    report = compare_economic_prefix(**kwargs)
    if not report["passed"]:
        failed = [
            label for label, frame in report["frames"].items() if not frame["passed"]
        ]
        raise EconomicPrefixError(
            "Published economic prefix changed in supposedly neutral migration: "
            + ", ".join(failed)
        )
    return report


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
        "captured_at_utc": timestamp.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "entrypoint": str(entrypoint),
        "command": {
            "argv": [str(value) for value in command_argv],
            "shell_escaped": shlex.join(str(value) for value in command_argv),
        },
        "git": {
            key: value for key, value in git.items() if key != "untracked_files"
        },
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
            errors.append(
                "git_dirty declaration does not match the current worktree"
            )
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


def _normalize_economic_frame(
    frame: pl.DataFrame,
    *,
    keys: tuple[str, ...],
    numeric_columns: tuple[str, ...],
    exact_candidates: tuple[str, ...],
    label: str,
) -> pl.DataFrame:
    required = set(keys) | set(numeric_columns)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    expressions: list[pl.Expr] = []
    for key in keys:
        if key.endswith("_month"):
            expressions.append(pl.col(key).cast(pl.Date, strict=False).alias(key))
        else:
            expressions.append(pl.col(key).cast(pl.String).alias(key))
    expressions.extend(
        pl.col(column).cast(pl.Float64, strict=False).alias(column)
        for column in numeric_columns
    )
    exact_columns = tuple(column for column in exact_candidates if column in frame.columns)
    expressions.extend(pl.col(column).alias(column) for column in exact_columns)
    normalized = frame.select(expressions).sort(keys)
    duplicate_count = normalized.height - normalized.select(
        pl.struct(keys).n_unique()
    ).item()
    if duplicate_count:
        raise ValueError(f"{label} has {duplicate_count} duplicate economic keys.")
    if normalized.select(pl.any_horizontal([pl.col(key).is_null() for key in keys])).to_series().any():
        raise ValueError(f"{label} contains null economic keys.")
    return normalized


def _compare_economic_frame(
    *,
    reference: pl.DataFrame,
    candidate: pl.DataFrame,
    keys: tuple[str, ...],
    numeric_columns: tuple[str, ...],
    exact_candidates: tuple[str, ...],
    tolerance: float,
) -> dict[str, Any]:
    reference_keys = reference.select(keys)
    candidate_keys = candidate.select(keys)
    missing_keys = reference_keys.join(candidate_keys, on=keys, how="anti")
    unexpected_keys = candidate_keys.join(reference_keys, on=keys, how="anti")
    exact_columns = tuple(
        column
        for column in exact_candidates
        if column in reference.columns and column in candidate.columns
    )
    missing_exact_columns = sorted(
        (set(exact_candidates) & set(reference.columns)) - set(candidate.columns)
    )
    joined = reference.join(candidate, on=keys, how="inner", suffix="_candidate")

    numeric_report: dict[str, Any] = {}
    for column in numeric_columns:
        left = pl.col(column)
        right = pl.col(f"{column}_candidate")
        null_mismatch = joined.filter(left.is_null() != right.is_null()).height
        finite_pairs = joined.filter(left.is_finite() & right.is_finite())
        nonfinite_mismatch = joined.filter(
            left.is_not_null()
            & right.is_not_null()
            & (~left.is_finite() | ~right.is_finite())
            & left.eq_missing(right).not_()
        ).height
        maximum = (
            finite_pairs.select((left - right).abs().max()).item()
            if not finite_pairs.is_empty()
            else 0.0
        )
        maximum = float(maximum or 0.0)
        numeric_report[column] = {
            "maximum_absolute_difference": maximum,
            "null_mismatches": null_mismatch,
            "nonfinite_mismatches": nonfinite_mismatch,
            "passed": null_mismatch == 0
            and nonfinite_mismatch == 0
            and maximum <= tolerance,
        }

    exact_report: dict[str, Any] = {}
    for column in exact_columns:
        mismatches = joined.filter(
            pl.col(column)
            .eq_missing(pl.col(f"{column}_candidate"))
            .not_()
        )
        exact_report[column] = {
            "mismatches": mismatches.height,
            "passed": mismatches.is_empty(),
        }
    for column in missing_exact_columns:
        exact_report[column] = {"mismatches": None, "passed": False, "missing": True}

    passed = (
        missing_keys.is_empty()
        and unexpected_keys.is_empty()
        and all(result["passed"] for result in numeric_report.values())
        and all(result["passed"] for result in exact_report.values())
    )
    return {
        "reference_rows": reference.height,
        "candidate_rows": candidate.height,
        "reference_sha256": _dataframe_sha256(reference),
        "candidate_sha256": _dataframe_sha256(candidate),
        "missing_keys": missing_keys.head(20).to_dicts(),
        "missing_key_count": missing_keys.height,
        "unexpected_keys": unexpected_keys.head(20).to_dicts(),
        "unexpected_key_count": unexpected_keys.height,
        "numeric_columns": numeric_report,
        "exact_columns": exact_report,
        "passed": passed,
    }


def _dataframe_sha256(frame: pl.DataFrame) -> str:
    buffer = BytesIO()
    frame.write_ipc(buffer, compression="uncompressed")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


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
        stderr = (
            result.stderr.decode("utf-8", errors="replace")
            if binary
            else result.stderr
        )
        raise RuntimeProvenanceError(
            f"Git command failed ({' '.join(args)}): {str(stderr).strip()}"
        )
    return result.stdout


def _capture_git_state(root: Path) -> dict[str, Any]:
    status = str(
        _git_command(root, "status", "--porcelain=v1", "--untracked-files=all")
    )
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
    untracked_output = str(
        _git_command(root, "ls-files", "--others", "--exclude-standard")
    )
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
        "branch": str(
            _git_command(root, "rev-parse", "--abbrev-ref", "HEAD")
        ).strip(),
        "dirty": bool(status),
        "status_porcelain_sha256": hashlib.sha256(
            status.encode("utf-8")
        ).hexdigest(),
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


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
            str(child_key): _sanitize_runtime_value(
                child_value, key=str(child_key)
            )
            for child_key, child_value in sorted(
                value.items(), key=lambda item: str(item[0])
            )
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


def _validate_sources(sources: Mapping[str, Path]) -> dict[str, Path]:
    normalized: dict[str, Path] = {}
    for raw_label, raw_path in sources.items():
        label = str(raw_label).strip()
        if not label or label in {".", ".."} or "/" in label or "\\" in label:
            raise ValueError(f"Invalid baseline source label: {raw_label!r}")
        source = Path(raw_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Baseline source does not exist: {source}")
        if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError(
                f"Baseline sources must not contain symlinks: {source}"
            )
        if label in normalized:
            raise ValueError(f"Duplicate baseline source label: {label}")
        normalized[label] = source
    return dict(sorted(normalized.items()))


def _files_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def _inventory_entry(
    path: Path, *, package_root: Path, storage_mode: str
) -> _InventoryEntry:
    return _InventoryEntry(
        relative_path=path.relative_to(package_root).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=_sha256_path(path),
        storage_mode=storage_mode,
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_sha256(entries: list[_InventoryEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item.relative_path):
        digest.update(entry.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry.size_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _storage_mode_counts(entries: list[_InventoryEntry]) -> dict[str, int]:
    return {
        mode: sum(entry.storage_mode == mode for entry in entries)
        for mode in sorted({entry.storage_mode for entry in entries})
    }


def _remove_write_bits(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(path.stat().st_mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)


def _make_tree_owner_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        try:
            path.chmod(path.stat().st_mode | 0o700)
        except OSError:
            continue
