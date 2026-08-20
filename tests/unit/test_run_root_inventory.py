from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpharank.governance_contracts.run_organization import (
    build_run_root_inventory,
    canonical_run_dir,
    initialize_run_manifest,
    transition_run_status,
    validate_canonical_run_dir,
    validate_run_manifest,
    validate_run_root_inventory,
    write_run_manifest,
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


def test_canonical_run_path_has_family_and_utc_run_id(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_dir = canonical_run_dir(
        outputs,
        family="monthly_legacy",
        run_id="20260820T191500Z_monthly",
    )

    report = validate_canonical_run_dir(outputs, run_dir)

    assert run_dir == outputs / "monthly_legacy" / "20260820T191500Z_monthly"
    assert report["contract"] == "alpharank_run_path_v1"


def test_canonical_run_path_rejects_free_form_or_nested_names(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"

    with pytest.raises(ValueError, match="Invalid run id"):
        canonical_run_dir(outputs, family="legacy", run_id="latest_final_v2")
    with pytest.raises(ValueError, match="exactly two parts"):
        validate_canonical_run_dir(
            outputs,
            outputs / "legacy" / "20260820T191500Z_monthly" / "retry",
        )


def test_run_manifest_starts_candidate_and_transitions_explicitly(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    manifest_path = initialize_run_manifest(
        outputs,
        family="monthly_legacy",
        run_id="20260820T191500Z_monthly",
        created_at="2026-08-20T19:15:00Z",
    )
    candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
    validated = transition_run_status(
        candidate,
        new_status="validated",
        changed_at="2026-08-20T19:20:00Z",
        reason="replay gates passed",
    )
    published = transition_run_status(
        validated,
        new_status="published",
        changed_at="2026-08-20T19:21:00Z",
        reason="owner promotion",
    )
    write_run_manifest(manifest_path, published)

    report = validate_run_manifest(outputs, manifest_path)

    assert report["status"] == "published"
    assert report["transition_count"] == 2


def test_run_manifest_rejects_invalid_status_transition(tmp_path: Path) -> None:
    manifest_path = initialize_run_manifest(
        tmp_path / "outputs",
        family="boosting_research",
        run_id="20260820T192000Z_challenger",
        created_at="2026-08-20T19:20:00Z",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="candidate -> published"):
        transition_run_status(
            manifest,
            new_status="published",
            changed_at="2026-08-20T19:21:00Z",
            reason="skip validation",
        )


def test_run_path_rejects_status_encoded_in_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="embeds manifest status"):
        canonical_run_dir(
            tmp_path / "outputs",
            family="monthly_legacy",
            run_id="20260820T192500Z_candidate",
        )
