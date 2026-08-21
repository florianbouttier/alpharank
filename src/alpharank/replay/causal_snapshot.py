"""Seal and validate immutable causal-v2 model-input packages."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from alpharank.data.publishing.composed_snapshot import validate_composed_model_snapshot
from alpharank.data.price_eligibility import (
    STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY,
)
from alpharank.data.prices.history import PERSISTENT_PRICE_HISTORY_POLICY_ID
from alpharank.data.publishing.snapshot_storage import copy_snapshot_file
from alpharank.governance import (
    capture_runtime_provenance,
    validate_runtime_provenance,
)
from alpharank.portfolio.execution import LEGACY_NEXT_SESSION_OPEN

CAUSAL_SNAPSHOT_CONTRACT_VERSION = 1
CAUSAL_SNAPSHOT_MANIFEST_NAME = "causal_v2_snapshot_manifest.json"
CAUSAL_SNAPSHOT_SEAL_NAME = "causal_v2_snapshot_manifest.sha256"
REQUIRED_POLICY_FILES = {
    "filing_availability": "configs/data_quality/filing_availability_policy_v1.json",
    "missing_fundamentals": "configs/data_quality/missing_fundamentals_policy_v1.json",
    "ticker_exclusions": "configs/data_quality/historical_ticker_exclusions_v1.json",
    "constituent_changes": "configs/data_quality/sp500_constituent_changes_2026.json",
    "corporate_actions": "configs/data_quality/confirmed_corporate_actions.json",
}
REQUIRED_CRITICAL_FILES = (
    "src/alpharank/causal_snapshot.py",
    "src/alpharank/replay/causal_snapshot.py",
    "src/alpharank/data/publishing/composed_snapshot.py",
    "src/alpharank/data/contracts/feature_availability.py",
    "src/alpharank/data/contracts/fundamental_coverage.py",
    "src/alpharank/data/price_eligibility.py",
    "src/alpharank/data/prices/history.py",
    "src/alpharank/data/contracts/sector_history.py",
    "src/alpharank/portfolio/execution.py",
    "src/alpharank/portfolio/simulation.py",
    "src/alpharank/portfolio/terminal_returns.py",
    "src/alpharank/governance.py",
    "src/alpharank/governance_contracts/common.py",
    "src/alpharank/governance_contracts/contracts.py",
    "src/alpharank/governance_contracts/runtime_provenance.py",
)


class CausalSnapshotValidationError(RuntimeError):
    """Raised when a causal-v2 input package is incomplete or modified."""


def seal_causal_v2_snapshot(
    *,
    source_snapshot_dir: Path,
    package_dir: Path,
    project_root: Path,
    command_argv: list[str] | tuple[str, ...],
    implementation_commit: str,
    sealed_at: datetime | None = None,
) -> dict[str, Any]:
    """Clone one validated composed snapshot and bind all causal policies to it."""

    source = source_snapshot_dir.resolve()
    destination = package_dir.resolve()
    root = project_root.resolve()
    if destination.exists():
        raise FileExistsError(f"Causal snapshot already exists: {destination}")
    source_validation = validate_composed_model_snapshot(source)
    source_manifest = _read_json(source / "snapshot_manifest.json")
    _require_production_source_contract(source, source_manifest)

    timestamp = sealed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("sealed_at must include an explicit timezone")
    snapshot_id = (
        "v2-causal-"
        f"{source_manifest['composition_id'][:12]}-"
        f"{timestamp.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    external_patch = Path(
        tempfile.NamedTemporaryFile(
            prefix="alpharank-causal-v2-git-state-",
            suffix=".json",
            delete=False,
        ).name
    )
    external_patch.unlink()
    try:
        runtime = capture_runtime_provenance(
            project_root=root,
            entrypoint="scripts/seal_causal_v2_snapshot.py",
            command_argv=command_argv,
            resolved_config={
                "causal_snapshot_contract_version": CAUSAL_SNAPSHOT_CONTRACT_VERSION,
                "source_snapshot_dir": str(source),
                "destination": str(destination),
                "sector_policy": "point_in_time_complete_or_cap_disabled",
                "missing_return_policy": "raise",
                "benchmark_price_column": "adjusted_close",
            },
            seeds={"snapshot_build": "deterministic_no_random_seed"},
            critical_files=list(REQUIRED_CRITICAL_FILES),
            data_identifiers={
                "composition_id": source_manifest["composition_id"],
                "price_run_id": source_manifest["source_packages"]["prices"]["run_id"],
                "sec_run_id": source_manifest["source_packages"]["sec"]["run_id"],
            },
            patch_path=external_patch,
            captured_at=timestamp,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        external_patch.unlink(missing_ok=True)
        raise
    temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=False, exist_ok=False)
    storage_modes: list[str] = []
    try:
        input_snapshot = temporary / "input_snapshot"
        for source_file in _files_under(source):
            target = input_snapshot / source_file.relative_to(source)
            storage_modes.append(copy_snapshot_file(source_file, target))

        policy_records: dict[str, dict[str, Any]] = {}
        for label, relative in REQUIRED_POLICY_FILES.items():
            policy_source = root / relative
            if not policy_source.is_file():
                raise FileNotFoundError(f"Required causal policy is missing: {relative}")
            policy_target = temporary / "policies" / Path(relative).name
            storage_modes.append(copy_snapshot_file(policy_source, policy_target))
            payload = _read_json(policy_target)
            policy_records[label] = {
                "relative_path": policy_target.relative_to(temporary).as_posix(),
                "sha256": _sha256(policy_target),
                "policy_id": payload.get("policy_id") or payload.get("registry_id"),
            }

        patch_path = temporary / "runtime" / "git_state.json"
        storage_modes.append(copy_snapshot_file(external_patch, patch_path))
        runtime["patch_artifact"]["path"] = str(
            destination / "runtime" / "git_state.json"
        )

        inventory = _inventory(temporary)
        manifest: dict[str, Any] = {
            "causal_snapshot_contract_version": CAUSAL_SNAPSHOT_CONTRACT_VERSION,
            "scope": "alpharank_causal_v2_snapshot",
            "snapshot_id": snapshot_id,
            "methodology_version": "v2-causal",
            "sealed_at_utc": timestamp.astimezone(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "implementation_commit": str(implementation_commit),
            "source_snapshot": {
                "path": str(source),
                "composition_id": source_manifest["composition_id"],
                "scope": source_manifest["scope"],
                "validation": source_validation,
                "source_packages": source_manifest["source_packages"],
                "data_freshness": source_manifest.get("data_freshness", {}),
            },
            "data_contract": {
                "fundamentals": "strict_SEC_only_available_at_decision",
                "allowed_fundamental_sources": [
                    "sec_companyfacts",
                    "sec_derived_eps",
                    "sec_filing",
                    "sec_submissions",
                ],
                "prices": "published_persistent_history_v1",
                "historical_universe": "point_in_time_membership",
                "sectors": {
                    "policy": "point_in_time_complete_or_cap_disabled",
                    "static_sector_fallback_allowed": False,
                    "source_artifact_present": False,
                    "missing_history_action": "disable_sector_cap",
                },
                "benchmark": {
                    "ticker": "SPY",
                    "price_column": "adjusted_close",
                    "forward_fill_allowed": False,
                },
            },
            "policies": {
                **policy_records,
                "persistent_price_history": {
                    "policy_id": PERSISTENT_PRICE_HISTORY_POLICY_ID,
                    "relative_path": (
                        "input_snapshot/lineage/prices/"
                        "persistent_price_history_registry.parquet"
                    ),
                    "sha256": _sha256(
                        input_snapshot
                        / "lineage"
                        / "prices"
                        / "persistent_price_history_registry.parquet"
                    ),
                },
                "monthly_price_eligibility": (
                    STANDARD_MONTHLY_PRICE_ELIGIBILITY_POLICY.to_manifest()
                ),
                "execution": LEGACY_NEXT_SESSION_OPEN.to_manifest(),
                "missing_selected_return": {"policy": "raise"},
                "terminal_return": {
                    "policy_version": 1,
                    "unresolved_action": "raise",
                },
            },
            "runtime_provenance": runtime,
            "storage_contract": {
                "strategy": "copy_on_write_with_physical_copy_fallback",
                "semantics": "independent byte-identical immutable payload paths",
                "storage_mode_counts": dict(sorted(Counter(storage_modes).items())),
            },
            "payload_file_count": len(inventory),
            "payload_size_bytes": sum(row["size_bytes"] for row in inventory),
            "payload_inventory_sha256": _canonical_sha256(inventory),
            "inventory": inventory,
        }
        manifest_path = temporary / CAUSAL_SNAPSHOT_MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / CAUSAL_SNAPSHOT_SEAL_NAME).write_text(
            f"{_sha256(manifest_path)}  {CAUSAL_SNAPSHOT_MANIFEST_NAME}\n",
            encoding="utf-8",
        )
        _remove_write_bits(temporary)
        os.replace(temporary, destination)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _make_tree_owner_writable(temporary)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        external_patch.unlink(missing_ok=True)

    return validate_causal_v2_snapshot(destination)


def validate_causal_v2_snapshot(package_dir: Path) -> dict[str, Any]:
    """Fail closed if a sealed causal-v2 package or policy differs by one byte."""

    root = package_dir.resolve()
    manifest_path = root / CAUSAL_SNAPSHOT_MANIFEST_NAME
    seal_path = root / CAUSAL_SNAPSHOT_SEAL_NAME
    errors: list[str] = []
    if not manifest_path.is_file() or not seal_path.is_file():
        raise CausalSnapshotValidationError("Causal snapshot manifest or seal is missing")
    expected_manifest_sha = seal_path.read_text(encoding="utf-8").split()[0]
    if _sha256(manifest_path) != expected_manifest_sha:
        errors.append("causal snapshot manifest SHA-256 mismatch")
    manifest = _read_json(manifest_path)
    if manifest.get("causal_snapshot_contract_version") != CAUSAL_SNAPSHOT_CONTRACT_VERSION:
        errors.append("unsupported causal snapshot contract version")
    if manifest.get("scope") != "alpharank_causal_v2_snapshot":
        errors.append("invalid causal snapshot scope")

    expected_inventory = manifest.get("inventory")
    if not isinstance(expected_inventory, list) or not expected_inventory:
        errors.append("causal snapshot inventory is missing")
        expected_inventory = []
    expected_by_path = {
        str(row.get("relative_path")): row
        for row in expected_inventory
        if isinstance(row, Mapping) and row.get("relative_path")
    }
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in _files_under(root)
        if path.name not in {CAUSAL_SNAPSHOT_MANIFEST_NAME, CAUSAL_SNAPSHOT_SEAL_NAME}
    }
    if set(expected_by_path) != actual_paths:
        errors.append("causal snapshot payload file set changed")
    actual_inventory: list[dict[str, Any]] = []
    for relative in sorted(set(expected_by_path) & actual_paths):
        path = root / relative
        expected = expected_by_path[relative]
        observed = {
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        actual_inventory.append(observed)
        if observed["sha256"] != expected.get("sha256"):
            errors.append(f"payload SHA-256 mismatch: {relative}")
        if observed["size_bytes"] != expected.get("size_bytes"):
            errors.append(f"payload size mismatch: {relative}")
    if _canonical_sha256(actual_inventory) != manifest.get("payload_inventory_sha256"):
        errors.append("payload inventory SHA-256 mismatch")

    try:
        composed = validate_composed_model_snapshot(root / "input_snapshot")
        if composed["composition_id"] != manifest.get("source_snapshot", {}).get(
            "composition_id"
        ):
            errors.append("source composition identity mismatch")
    except (KeyError, RuntimeError, FileNotFoundError) as exc:
        errors.append(f"invalid composed input snapshot: {exc}")

    data_contract = manifest.get("data_contract", {})
    sectors = data_contract.get("sectors", {}) if isinstance(data_contract, Mapping) else {}
    if sectors.get("static_sector_fallback_allowed") is not False:
        errors.append("static sector fallback must be forbidden")
    if sectors.get("missing_history_action") != "disable_sector_cap":
        errors.append("missing PIT sector history must disable the sector cap")
    policies = manifest.get("policies", {})
    if (
        policies.get("persistent_price_history", {}).get("policy_id")
        != PERSISTENT_PRICE_HISTORY_POLICY_ID
    ):
        errors.append("persistent price history policy is missing")
    if policies.get("filing_availability", {}).get("policy_id") != "sec-filing-availability-v1":
        errors.append("filing availability policy is missing")
    if policies.get("missing_fundamentals", {}).get("policy_id") != "sec-only-exclude-ex-ante-v1":
        errors.append("missing-fundamentals policy is missing")
    if policies.get("execution", {}).get("identifier") != "next_session_open_v1":
        errors.append("canonical execution policy is missing")
    if policies.get("missing_selected_return", {}).get("policy") != "raise":
        errors.append("missing selected returns do not fail closed")
    try:
        validate_runtime_provenance(manifest.get("runtime_provenance", {}))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"invalid runtime provenance: {exc}")

    if errors:
        raise CausalSnapshotValidationError("; ".join(errors))
    return {
        "passed": True,
        "snapshot_id": manifest["snapshot_id"],
        "composition_id": manifest["source_snapshot"]["composition_id"],
        "manifest_sha256": _sha256(manifest_path),
        "payload_inventory_sha256": manifest["payload_inventory_sha256"],
        "payload_file_count": len(actual_inventory),
        "package_dir": str(root),
    }


def _require_production_source_contract(source: Path, manifest: Mapping[str, Any]) -> None:
    validation = manifest.get("validation", {})
    if manifest.get("scope") != "alpharank_composed_model_input":
        raise CausalSnapshotValidationError("Source is not a composed model snapshot")
    if validation.get("passed") is not True:
        raise CausalSnapshotValidationError("Source composition is not validated")
    if validation.get("fundamental_contract") != "strict SEC-only":
        raise CausalSnapshotValidationError("Source fundamentals are not strict SEC-only")
    if validation.get("same_snapshot_for_legacy_and_boosting") is not True:
        raise CausalSnapshotValidationError("Source is not shared by Legacy and Boosting")
    if validation.get("persistent_price_registry_copied") is not True:
        raise CausalSnapshotValidationError("Source has no persistent price registry")
    price_manifest = _read_json(source / "lineage" / "prices" / "manifest.json")
    if int(price_manifest.get("contract_version", 0)) < 2:
        raise CausalSnapshotValidationError("Price lineage predates persistent-history v2")
    persistence = price_manifest.get("source_refresh_contract", {}).get(
        "persistent_price_history", {}
    )
    if persistence.get("policy_id") != PERSISTENT_PRICE_HISTORY_POLICY_ID:
        raise CausalSnapshotValidationError("Invalid persistent price history policy")
    if persistence.get("routine_deletion_allowed") is not False:
        raise CausalSnapshotValidationError("Persistent price history allows deletion")


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _files_under(root)
    ]


def _files_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CausalSnapshotValidationError(f"Expected a JSON object: {path}")
    return payload


def _remove_write_bits(root: Path) -> None:
    directories = sorted(path for path in root.rglob("*") if path.is_dir())
    for path in [*_files_under(root), *directories, root]:
        path.chmod(path.stat().st_mode & ~0o222)


def _make_tree_owner_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        try:
            path.chmod(path.stat().st_mode | 0o200)
        except FileNotFoundError:
            pass
