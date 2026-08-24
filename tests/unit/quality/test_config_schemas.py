from __future__ import annotations

from pathlib import Path

from alpharank.quality.config_schemas import (
    infer_structural_schema,
    validate_config_schema_registry,
    validate_config_value,
)

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "configs/data_contracts/config_schema_registry_v1.json"


def test_every_repository_json_config_has_a_strict_versioned_schema() -> None:
    report = validate_config_schema_registry(ROOT, REGISTRY)

    assert report["passed"] is True
    assert report["family_count"] == 17
    assert report["config_file_count"] == 21


def test_structural_schema_rejects_unknown_nested_key() -> None:
    schema = infer_structural_schema(
        [
            {
                "policy_id": "fixture_v1",
                "settings": {"threshold": 1.0, "enabled": True},
            }
        ]
    )
    contaminated = {
        "policy_id": "fixture_v1",
        "settings": {"threshold": 1.0, "enabled": True, "threshhold": 2.0},
    }

    errors = validate_config_value(contaminated, schema)

    assert errors == ["$.settings: unknown key threshhold"]
