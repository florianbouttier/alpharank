"""Versioned structural schemas for every maintained JSON config family."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Mapping, Sequence

CONFIG_SCHEMA_REGISTRY_VERSION = 1
CONFIG_FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "raw_provider_contracts",
        ("configs/data_contracts/raw_provider_contracts_v1.json",),
    ),
    (
        "confirmed_corporate_actions",
        ("configs/data_quality/confirmed_corporate_actions.json",),
    ),
    (
        "reviewed_extreme_price_moves",
        ("configs/data_quality/reviewed_extreme_price_moves.json",),
    ),
    (
        "filing_availability_policy",
        ("configs/data_quality/filing_availability_policy_v1.json",),
    ),
    (
        "historical_ticker_exclusions",
        ("configs/data_quality/historical_ticker_exclusions_v1.json",),
    ),
    (
        "missing_fundamentals_policy",
        ("configs/data_quality/missing_fundamentals_policy_v1.json",),
    ),
    (
        "sp500_constituent_changes",
        ("configs/data_quality/sp500_constituent_changes_2026.json",),
    ),
    (
        "terminal_shareholder_events",
        ("configs/data_quality/terminal_shareholder_events_v*.json",),
    ),
    (
        "terminal_successor_prices",
        ("configs/data_quality/terminal_successor_prices_v1.json",),
    ),
    ("ruff_baseline", ("configs/quality/ruff_baseline_v1.json",)),
    (
        "python_size_baseline",
        ("configs/quality/python_size_baseline_v1.json",),
    ),
    (
        "python_directory_policy",
        ("configs/quality/python_directory_policy_v1.json",),
    ),
    ("pytest_suite_policy", ("configs/quality/test_suites_v1.json",)),
    (
        "approved_terminal_target_censoring",
        ("configs/research/approved_terminal_target_censoring_v1.json",),
    ),
    (
        "legacy_ema_risk_overlay",
        ("configs/research/legacy_ema_risk_overlay_*.json",),
    ),
    (
        "legacy_ema_top_n_comparison",
        ("configs/research/legacy_ema_top5_vs_top10_quarantine_v7.json",),
    ),
    (
        "locked_legacy_ema_challenger",
        ("configs/research/locked_legacy_ema_challenger_v1.json",),
    ),
)
CONFIG_SEARCH_ROOTS = (
    "configs/data_contracts",
    "configs/data_quality",
    "configs/quality",
    "configs/research",
)


def build_config_schema_registry(root: Path) -> dict[str, object]:
    """Infer one strict versioned schema per reviewed config family."""

    discovered, discovery_errors = _discover_config_files(root)
    if discovery_errors:
        raise ValueError("; ".join(discovery_errors))
    families = []
    for family_id, patterns in CONFIG_FAMILY_PATTERNS:
        paths = discovered[family_id]
        values = [_load_json(path) for path in paths]
        schema = infer_structural_schema(values)
        if family_id == "ruff_baseline":
            _allow_dynamic_integer_map(schema, "diagnostics_by_code")
            _allow_dynamic_integer_map(schema, "diagnostics_by_path")
        families.append(
            {
                "family_id": family_id,
                "patterns": list(patterns),
                "example_files": [path.relative_to(root).as_posix() for path in paths],
                "schema": schema,
            }
        )
    return {
        "schema_version": CONFIG_SCHEMA_REGISTRY_VERSION,
        "registry_id": "alpharank_json_config_schemas_v1",
        "description": "Strict structural schemas inferred once from reviewed configuration families.",
        "families": families,
    }


def validate_config_schema_registry(root: Path, registry_path: Path) -> dict[str, object]:
    """Validate classification and contents of every maintained JSON config."""

    registry = _require_mapping(_load_json(registry_path), "schema registry")
    expected_registry_keys = {"schema_version", "registry_id", "description", "families"}
    errors = _key_errors(registry, expected_registry_keys, expected_registry_keys, "registry")
    if registry.get("schema_version") != CONFIG_SCHEMA_REGISTRY_VERSION:
        errors.append("registry.schema_version is unsupported")
    raw_families = registry.get("families")
    if not isinstance(raw_families, list):
        errors.append("registry.families must be a list")
        raw_families = []
    registry_families: dict[str, Mapping[str, object]] = {}
    for index, raw_family in enumerate(raw_families):
        try:
            family = _require_mapping(raw_family, f"registry.families[{index}]")
            family_id = _require_string(family.get("family_id"), "family_id")
        except ValueError as error:
            errors.append(str(error))
            continue
        family_keys = {"family_id", "patterns", "example_files", "schema"}
        errors.extend(_key_errors(family, family_keys, family_keys, f"family[{family_id}]"))
        if family_id in registry_families:
            errors.append(f"duplicate schema family: {family_id}")
        registry_families[family_id] = family

    expected_patterns = dict(CONFIG_FAMILY_PATTERNS)
    if set(registry_families) != set(expected_patterns):
        errors.append(
            "schema families differ: "
            f"missing={sorted(set(expected_patterns) - set(registry_families))}, "
            f"unknown={sorted(set(registry_families) - set(expected_patterns))}"
        )
    discovered, discovery_errors = _discover_config_files(root)
    errors.extend(discovery_errors)
    family_reports: list[dict[str, object]] = []
    for family_id, patterns in CONFIG_FAMILY_PATTERNS:
        current_family = registry_families.get(family_id)
        paths = discovered.get(family_id, [])
        family_errors: list[str] = []
        if current_family is None:
            family_errors.append(f"missing schema family: {family_id}")
        else:
            observed_patterns = current_family.get("patterns")
            if observed_patterns != list(patterns):
                family_errors.append(f"{family_id}: patterns differ from the versioned classifier")
            schema = current_family.get("schema")
            if not isinstance(schema, dict):
                family_errors.append(f"{family_id}: schema must be an object")
            else:
                for path in paths:
                    family_errors.extend(
                        validate_config_value(
                            _load_json(path),
                            schema,
                            path=path.relative_to(root).as_posix(),
                        )
                    )
        errors.extend(family_errors)
        family_reports.append(
            {
                "family_id": family_id,
                "file_count": len(paths),
                "passed": not family_errors,
                "errors": family_errors,
            }
        )
    return {
        "registry_id": registry.get("registry_id"),
        "schema_version": registry.get("schema_version"),
        "passed": not errors,
        "family_count": len(CONFIG_FAMILY_PATTERNS),
        "config_file_count": sum(len(paths) for paths in discovered.values()),
        "errors": errors,
        "families": family_reports,
    }


def infer_structural_schema(values: Sequence[object]) -> dict[str, object]:
    """Infer a deterministic strict schema from one or more reviewed examples."""

    if not values:
        return {"type": "any"}
    kinds = {_value_kind(value) for value in values}
    if kinds <= {"integer", "number"}:
        return {"type": "number" if "number" in kinds else "integer"}
    if len(kinds) > 1:
        variants = [
            infer_structural_schema([value for value in values if _value_kind(value) == kind])
            for kind in sorted(kinds)
        ]
        return {"any_of": variants}
    kind = next(iter(kinds))
    if kind == "object":
        mappings = [_require_mapping(value, "schema example") for value in values]
        all_keys = sorted({key for mapping in mappings for key in mapping})
        required = sorted(set.intersection(*(set(mapping) for mapping in mappings)))
        properties = {
            key: infer_structural_schema([mapping[key] for mapping in mappings if key in mapping])
            for key in all_keys
        }
        return {
            "type": "object",
            "required": required,
            "properties": properties,
            "additional_properties": False,
        }
    if kind == "array":
        items = [item for value in values if isinstance(value, list) for item in value]
        return {"type": "array", "items": infer_structural_schema(items)}
    return {"type": kind}


def validate_config_value(
    value: object,
    schema: Mapping[str, object],
    *,
    path: str = "$",
) -> list[str]:
    """Validate a config value and reject every undeclared object key."""

    variants = schema.get("any_of")
    if isinstance(variants, list):
        candidate_errors = [
            validate_config_value(value, _require_mapping(variant, "schema variant"), path=path)
            for variant in variants
        ]
        return [] if any(not errors for errors in candidate_errors) else [f"{path}: no schema matched"]
    expected = schema.get("type")
    if expected == "any":
        return []
    if expected == "object":
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            return [f"{path}: expected object"]
        mapping = _require_mapping(value, path)
        properties = _require_mapping(schema.get("properties"), f"{path} properties")
        required_raw = schema.get("required")
        required = set(required_raw) if isinstance(required_raw, list) else set()
        errors = [f"{path}: missing key {key}" for key in sorted(required - set(mapping))]
        additional = schema.get("additional_properties", False)
        for key in sorted(mapping):
            child_path = f"{path}.{key}"
            if key in properties:
                child_schema = _require_mapping(properties[key], f"{child_path} schema")
                errors.extend(validate_config_value(mapping[key], child_schema, path=child_path))
            elif additional is False:
                errors.append(f"{path}: unknown key {key}")
            elif isinstance(additional, dict):
                errors.extend(validate_config_value(mapping[key], additional, path=child_path))
        return errors
    if expected == "array":
        if not isinstance(value, list):
            return [f"{path}: expected array"]
        item_schema = _require_mapping(schema.get("items"), f"{path} item schema")
        return [
            error
            for index, item in enumerate(value)
            for error in validate_config_value(item, item_schema, path=f"{path}[{index}]")
        ]
    if not _matches_scalar_type(value, expected):
        return [f"{path}: expected {expected}, observed {_value_kind(value)}"]
    return []


def write_config_schema_registry(path: Path, registry: Mapping[str, object]) -> None:
    """Write the deterministic versioned registry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _discover_config_files(root: Path) -> tuple[dict[str, list[Path]], list[str]]:
    discovered: dict[str, list[Path]] = {
        family_id: [] for family_id, _ in CONFIG_FAMILY_PATTERNS
    }
    errors: list[str] = []
    for search_root in CONFIG_SEARCH_ROOTS:
        for path in sorted((root / search_root).glob("*.json")):
            relative = path.relative_to(root).as_posix()
            if relative == "configs/data_contracts/config_schema_registry_v1.json":
                continue
            matches = [
                family_id
                for family_id, patterns in CONFIG_FAMILY_PATTERNS
                if any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns)
            ]
            if len(matches) != 1:
                errors.append(f"{relative}: expected one config family, observed {matches}")
                continue
            discovered[matches[0]].append(path)
    for family_id, paths in discovered.items():
        if not paths:
            errors.append(f"config family has no files: {family_id}")
    return discovered, errors


def _allow_dynamic_integer_map(schema: dict[str, object], property_name: str) -> None:
    properties = _require_mapping(schema.get("properties"), "root schema properties")
    mutable_properties = dict(properties)
    mutable_properties[property_name] = {
        "type": "object",
        "required": [],
        "properties": {},
        "additional_properties": {"type": "integer"},
    }
    schema["properties"] = mutable_properties


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _key_errors(
    value: Mapping[str, object],
    allowed: set[str],
    required: set[str],
    path: str,
) -> list[str]:
    return [
        *[f"{path}: unknown key {key}" for key in sorted(set(value) - allowed)],
        *[f"{path}: missing key {key}" for key in sorted(required - set(value))],
    ]


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _value_kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    raise ValueError(f"Unsupported JSON value type: {type(value).__name__}")


def _matches_scalar_type(value: object, expected: object) -> bool:
    observed = _value_kind(value)
    return observed == expected or (expected == "number" and observed == "integer")
