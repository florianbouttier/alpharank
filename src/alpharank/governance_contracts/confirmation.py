"""Single-use sealed confirmation governance contract."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from alpharank.governance_contracts.common import (
    atomic_replace_json as _atomic_replace_json,
)
from alpharank.governance_contracts.common import (
    canonical_json_sha256 as _canonical_json_sha256,
)
from alpharank.governance_contracts.common import (
    directory_hashes as _directory_hashes,
)
from alpharank.governance_contracts.common import (
    promotion_timestamp as _promotion_timestamp,
)
from alpharank.governance_contracts.contracts import (
    SEALED_CONFIRMATION_CONTRACT_VERSION,
    SealedConfirmationError,
)


def create_sealed_confirmation_protocol(
    *,
    registry_path: Path,
    dataset_dir: Path,
    period_id: str,
    period_start: str,
    period_end: str,
    expected_experiment_ids: tuple[str, ...],
    approved_by: str,
    sealed_at: datetime | None = None,
) -> dict[str, Any]:
    """Declare the final period and complete experiment list before observation."""

    destination = registry_path.resolve()
    if destination.exists():
        raise FileExistsError(f"Confirmation registry already exists: {destination}")
    experiment_ids = tuple(str(value).strip() for value in expected_experiment_ids)
    if not experiment_ids or any(not value for value in experiment_ids):
        raise ValueError("expected_experiment_ids must contain non-empty identifiers.")
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("expected_experiment_ids must be unique.")
    dataset = dataset_dir.resolve()
    hashes = _directory_hashes(dataset)
    if not hashes:
        raise ValueError("The sealed confirmation dataset must contain files.")
    payload = {
        "confirmation_contract_version": SEALED_CONFIRMATION_CONTRACT_VERSION,
        "period_id": str(period_id).strip(),
        "period_start": str(period_start),
        "period_end": str(period_end),
        "dataset_dir": str(dataset),
        "dataset_hashes": hashes,
        "expected_experiment_ids": list(experiment_ids),
        "experiments": [],
        "status": "sealed",
        "approved_by": str(approved_by),
        "sealed_at_utc": _promotion_timestamp(sealed_at),
        "opened_at_utc": None,
        "opened_by": None,
        "open_reason": None,
        "experiments_inventory_sha256": None,
        "invalidations": [],
    }
    if not payload["period_id"]:
        raise ValueError("period_id must be non-empty.")
    _atomic_replace_json(destination, payload)
    return payload


def register_confirmation_experiment(
    *,
    registry_path: Path,
    experiment_id: str,
    hypothesis: str,
    command: str,
    config_sha256: str,
    result_manifest_sha256: str,
    registered_at: datetime | None = None,
) -> dict[str, Any]:
    """Register one completed variant before the sealed period is opened."""

    registry = _read_confirmation_registry(registry_path)
    identifier = str(experiment_id).strip()
    if registry["status"] != "sealed":
        _invalidate_confirmation(
            registry_path,
            registry,
            reason=f"optimization_after_{registry['status']}",
            changed_at=registered_at,
        )
        raise SealedConfirmationError(
            "Experiments cannot be registered after confirmation opening."
        )
    expected = set(registry["expected_experiment_ids"])
    if identifier not in expected:
        raise ValueError(f"Experiment was not declared before sealing: {identifier}")
    experiments = list(registry["experiments"])
    if identifier in {item["experiment_id"] for item in experiments}:
        raise ValueError(f"Experiment is already registered: {identifier}")
    required = {
        "hypothesis": str(hypothesis).strip(),
        "command": str(command).strip(),
        "config_sha256": str(config_sha256).strip(),
        "result_manifest_sha256": str(result_manifest_sha256).strip(),
    }
    if any(not value for value in required.values()):
        raise ValueError("Every experiment must record hypothesis, command and hashes.")
    experiments.append(
        {
            "experiment_id": identifier,
            **required,
            "registered_at_utc": _promotion_timestamp(registered_at),
        }
    )
    updated = {**registry, "experiments": experiments}
    _atomic_replace_json(registry_path, updated)
    return updated


def open_sealed_confirmation(
    *,
    registry_path: Path,
    opened_by: str,
    reason: str,
    opened_at: datetime | None = None,
) -> dict[str, Any]:
    """Open the final period once, only after the declared registry is complete."""

    registry = _read_confirmation_registry(registry_path)
    if registry["status"] != "sealed":
        _invalidate_confirmation(
            registry_path,
            registry,
            reason=f"reopen_attempt_after_{registry['status']}",
            changed_at=opened_at,
        )
        raise SealedConfirmationError("The sealed confirmation period is single-use.")
    expected = set(registry["expected_experiment_ids"])
    registered = {item["experiment_id"] for item in registry["experiments"]}
    if registered != expected:
        missing = sorted(expected - registered)
        _invalidate_confirmation(
            registry_path,
            registry,
            reason=f"premature_open_missing_experiments:{','.join(missing)}",
            changed_at=opened_at,
        )
        raise SealedConfirmationError(
            f"Confirmation opened before the experiment registry was complete: {missing}"
        )
    _validate_confirmation_dataset(registry)
    inventory_hash = _canonical_json_sha256(registry["experiments"])
    updated = {
        **registry,
        "status": "opened",
        "opened_at_utc": _promotion_timestamp(opened_at),
        "opened_by": str(opened_by),
        "open_reason": str(reason),
        "experiments_inventory_sha256": inventory_hash,
    }
    _atomic_replace_json(registry_path, updated)
    return updated


def validate_confirmation_for_promotion(registry_path: Path) -> dict[str, Any]:
    """Fail closed unless the single-use confirmation remains intact."""

    registry = _read_confirmation_registry(registry_path)
    if registry["status"] != "opened":
        raise SealedConfirmationError(
            f"Confirmation is not promotion-eligible: status={registry['status']}"
        )
    _validate_confirmation_dataset(registry)
    actual = _canonical_json_sha256(registry["experiments"])
    if actual != registry.get("experiments_inventory_sha256"):
        raise SealedConfirmationError("The experiment registry changed after opening.")
    return registry


def _read_confirmation_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Confirmation registry not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("confirmation_contract_version") != SEALED_CONFIRMATION_CONTRACT_VERSION:
        raise SealedConfirmationError("Unsupported sealed confirmation contract.")
    return payload


def _validate_confirmation_dataset(registry: Mapping[str, Any]) -> None:
    dataset_dir = Path(str(registry["dataset_dir"]))
    actual = _directory_hashes(dataset_dir)
    if actual != registry.get("dataset_hashes"):
        raise SealedConfirmationError("The sealed confirmation dataset was modified.")


def _invalidate_confirmation(
    path: Path,
    registry: Mapping[str, Any],
    *,
    reason: str,
    changed_at: datetime | None,
) -> None:
    invalidations = list(registry.get("invalidations") or [])
    invalidations.append(
        {
            "reason": str(reason),
            "changed_at_utc": _promotion_timestamp(changed_at),
        }
    )
    _atomic_replace_json(
        path,
        {**registry, "status": "invalidated", "invalidations": invalidations},
    )
