from __future__ import annotations

import json
from pathlib import Path

from alpharank.quality.data_locations import validate_data_location_inventory

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs" / "architecture" / "data_location_inventory_v1.json"


def test_data_location_inventory_declares_current_files_packages_and_readers() -> None:
    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))

    report = validate_data_location_inventory(ROOT, payload)

    assert report["passed"] is True, report["errors"]
    assert payload["summary"]["file_location_count"] == 10
    assert payload["summary"]["package_location_count"] == 25
    assert payload["summary"]["reader_edge_count"] > 0


def test_canonical_pointer_and_legacy_prices_have_explicit_readers() -> None:
    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    by_id = {row["location_id"]: row for row in payload["locations"]}

    assert by_id["legacy_prices"]["readers"]
    assert by_id["model_input_manifests"]["readers"]
    assert by_id["warehouse_mart"]["migration_status"] == "target"
