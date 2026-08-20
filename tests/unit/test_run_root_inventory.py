from __future__ import annotations

import json
from pathlib import Path

from alpharank.governance_contracts.run_organization import (
    build_run_root_inventory,
    validate_run_root_inventory,
)


def test_run_root_inventory_exposes_family_date_status_and_size(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    published = outputs / "production_refresh_20260820"
    published.mkdir(parents=True)
    (published / "result.parquet").write_bytes(b"result")
    (published / "manifest.json").write_text(
        json.dumps({"status": "published"}),
        encoding="utf-8",
    )
    legacy = outputs / "legacy_probe_20260727"
    legacy.mkdir()
    (legacy / "evidence.txt").write_text("evidence", encoding="utf-8")

    inventory = build_run_root_inventory(outputs, observed_at="2026-08-20")
    report = validate_run_root_inventory(outputs, inventory)
    by_name = {row["root_name"]: row for row in inventory["run_roots"]}

    assert report["run_root_count"] == 2
    assert by_name["production_refresh_20260820"]["family"] == "production_refresh"
    assert by_name["production_refresh_20260820"]["run_date"] == "2026-08-20"
    assert by_name["production_refresh_20260820"]["status"] == "published"
    assert by_name["legacy_probe_20260727"]["status"] == "legacy_unclassified"
    assert report["size_bytes"] > 0
