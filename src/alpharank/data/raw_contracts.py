"""Semantic validation and lookup for RAW provider contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

RAW_PROVIDER_CONTRACT_ID = "alpharank_raw_provider_contracts_v1"
RAW_RECEIPT_CONTRACT_ID = "alpharank_raw_receipt_v1"
RAW_PROVIDER_MANIFEST_CONTRACT_ID = "alpharank_raw_provider_manifest_v1"
DEFAULT_RAW_PROVIDER_CONTRACTS = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "data_contracts"
    / "raw_provider_contracts_v1.json"
)


def load_raw_provider_contracts(
    path: Path = DEFAULT_RAW_PROVIDER_CONTRACTS,
) -> dict[str, object]:
    """Load the provider registry and reject unsafe RAW destinations."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RAW provider contracts must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported RAW provider contract schema")
    if payload.get("contract_id") != RAW_PROVIDER_CONTRACT_ID:
        raise ValueError("Unsupported RAW provider contract id")
    if payload.get("manifest_contract") != RAW_PROVIDER_MANIFEST_CONTRACT_ID:
        raise ValueError("Unsupported RAW provider manifest contract")
    manifest_fields = payload.get("provider_manifest_required_fields")
    if not isinstance(manifest_fields, list) or len(manifest_fields) != len(
        set(map(str, manifest_fields))
    ):
        raise ValueError("RAW provider manifest fields must be a unique list")
    receipt = _mapping(payload.get("receipt_contract"), "receipt_contract")
    if receipt.get("contract_id") != RAW_RECEIPT_CONTRACT_ID:
        raise ValueError("Unsupported RAW receipt contract")
    required_fields = receipt.get("required_fields")
    if not isinstance(required_fields, list) or len(required_fields) != len(
        set(map(str, required_fields))
    ):
        raise ValueError("RAW receipt required_fields must be a unique list")

    providers = payload.get("providers")
    if not isinstance(providers, list) or not providers:
        raise ValueError("RAW provider contracts require providers")
    provider_ids: set[str] = set()
    target_roots: set[str] = set()
    for raw_provider in providers:
        provider = _mapping(raw_provider, "provider")
        provider_id = _string(provider.get("provider_id"), "provider_id")
        target_root = _string(provider.get("target_root"), "target_root")
        target_manifest = _string(
            provider.get("target_manifest_path"), "target_manifest_path"
        )
        if provider_id in provider_ids:
            raise ValueError(f"Duplicate RAW provider id: {provider_id}")
        if target_root in target_roots:
            raise ValueError(f"Duplicate RAW provider target: {target_root}")
        if target_root != f"data/warehouse/raw/{provider_id}":
            raise ValueError(f"RAW provider target is not canonical: {provider_id}")
        if not target_manifest.startswith(f"{target_root}/manifests/"):
            raise ValueError(f"RAW provider manifest is outside target: {provider_id}")
        if provider.get("migration_status") == "published":
            raise ValueError("RAW provider contracts cannot publish model inputs")
        datasets = provider.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError(f"RAW provider has no datasets: {provider_id}")
        for raw_dataset in datasets:
            dataset = _mapping(raw_dataset, f"dataset for {provider_id}")
            for field in ("business_key", "formats", "request_identity_fields"):
                values = dataset.get(field)
                if not isinstance(values, list) or not values:
                    raise ValueError(f"{provider_id} dataset requires {field}")
        if provider.get("migration_status") == "catalogued_by_hash":
            evidence = _mapping(provider.get("catalog_evidence"), "catalog_evidence")
            if not isinstance(evidence.get("source_file_count"), int):
                raise ValueError(f"{provider_id} catalog evidence requires source count")
            digest = evidence.get("catalog_manifest_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{provider_id} catalog evidence requires SHA-256")
        provider_ids.add(provider_id)
        target_roots.add(target_root)
    return payload


def provider_contract(
    provider_id: str,
    *,
    contracts: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Resolve exactly one declared provider or fail closed."""

    payload = contracts or load_raw_provider_contracts()
    providers = payload.get("providers")
    if not isinstance(providers, list):
        raise ValueError("RAW provider contracts require providers")
    matches = [
        _mapping(provider, "provider")
        for provider in providers
        if isinstance(provider, Mapping) and provider.get("provider_id") == provider_id
    ]
    if len(matches) != 1:
        raise KeyError(f"Unknown RAW provider: {provider_id}")
    return matches[0]


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value
