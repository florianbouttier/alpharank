from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from alpharank.replay.causal_snapshot import (
    REQUIRED_CRITICAL_FILES,
    REQUIRED_POLICY_FILES,
    CausalSnapshotValidationError,
    seal_causal_v2_snapshot,
    validate_causal_v2_snapshot,
)


def test_causal_v2_snapshot_is_sealed_and_complete(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "source"
    package = project / "sealed"
    _make_git_project(project)
    _make_composed_snapshot(source)
    result = seal_causal_v2_snapshot(
        source_snapshot_dir=source,
        package_dir=package,
        project_root=project,
        command_argv=["python", "scripts/seal_causal_v2_snapshot.py"],
        implementation_commit=_git(project, "rev-parse", "HEAD"),
        sealed_at=datetime(2026, 8, 18, 12, tzinfo=timezone.utc),
    )

    manifest = json.loads(
        (package / "causal_v2_snapshot_manifest.json").read_text(encoding="utf-8")
    )
    assert result["passed"] is True
    assert manifest["scope"] == "alpharank_causal_v2_snapshot"
    assert manifest["source_snapshot"]["scope"] == "alpharank_composed_model_input"
    assert manifest["data_contract"]["fundamentals"] == "strict_SEC_only_available_at_decision"
    assert manifest["data_contract"]["sectors"] == {
        "missing_history_action": "disable_sector_cap",
        "policy": "point_in_time_complete_or_cap_disabled",
        "source_artifact_present": False,
        "static_sector_fallback_allowed": False,
    }
    assert (
        manifest["policies"]["persistent_price_history"]["policy_id"]
        == "published_price_history_v1"
    )
    assert manifest["policies"]["filing_availability"]["policy_id"] == "sec-filing-availability-v1"
    assert (
        manifest["policies"]["missing_fundamentals"]["policy_id"]
        == "sec-only-exclude-ex-ante-v1"
    )
    assert manifest["policies"]["execution"]["identifier"] == "next_session_open_v1"
    assert manifest["policies"]["missing_selected_return"]["policy"] == "raise"
    assert (
        package
        / "input_snapshot/lineage/prices/persistent_price_history_registry.parquet"
    ).is_file()
    assert result["payload_file_count"] == manifest["payload_file_count"]

    price = package / "input_snapshot" / "US_Finalprice.parquet"
    price.chmod(price.stat().st_mode | 0o200)
    price.write_bytes(price.read_bytes() + b"mutation")
    with pytest.raises(CausalSnapshotValidationError, match="payload SHA-256 mismatch"):
        validate_causal_v2_snapshot(package)


def _make_git_project(project: Path) -> None:
    project.mkdir()
    for relative in REQUIRED_CRITICAL_FILES:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    for label, relative in REQUIRED_POLICY_FILES.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"policy_id": f"{label}-v1"}
        if label == "filing_availability":
            payload["policy_id"] = "sec-filing-availability-v1"
        elif label == "missing_fundamentals":
            payload["policy_id"] = "sec-only-exclude-ex-ante-v1"
        path.write_text(json.dumps(payload), encoding="utf-8")
    _git(project, "init")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Test")
    _git(project, "add", ".")
    _git(project, "commit", "-m", "fixture")


def _make_composed_snapshot(source: Path) -> None:
    source.mkdir(parents=True)
    payloads = {
        "US_Finalprice.parquet": pl.DataFrame({"ticker": ["A.US"], "date": ["2026-08-14"]}),
        "SP500Price.parquet": pl.DataFrame({"date": ["2026-08-14"], "adjusted_close": [100.0]}),
    }
    for name, frame in payloads.items():
        frame.write_parquet(source / name)
    (source / "SP500_Constituents.csv").write_text(
        "Date,Ticker,Name\n2026-08-01,A,Alpha\n", encoding="utf-8"
    )
    for name in (
        "US_General.parquet",
        "US_Income_statement.parquet",
        "US_Balance_sheet.parquet",
        "US_Cash_flow.parquet",
        "US_Earnings.parquet",
    ):
        pl.DataFrame({"ticker": ["A.US"]}).write_parquet(source / name)
    price_lineage = source / "lineage" / "prices"
    price_lineage.mkdir(parents=True)
    pl.DataFrame({"ticker": ["A.US"]}).write_parquet(
        price_lineage / "persistent_price_history_registry.parquet"
    )
    price_manifest = {
        "contract_version": 2,
        "source_refresh_contract": {
            "persistent_price_history": {
                "policy_id": "published_price_history_v1",
                "routine_deletion_allowed": False,
            }
        },
    }
    (price_lineage / "manifest.json").write_text(json.dumps(price_manifest), encoding="utf-8")
    output_sha = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.iterdir()
        if path.is_file() and path.name != "snapshot_manifest.json"
    }
    manifest = {
        "scope": "alpharank_composed_model_input",
        "composition_id": "a" * 64,
        "source_packages": {
            "prices": {"run_id": "prices-v1"},
            "sec": {"run_id": "sec-v1"},
        },
        "output_sha256": output_sha,
        "validation": {
            "passed": True,
            "fundamental_contract": "strict SEC-only",
            "same_snapshot_for_legacy_and_boosting": True,
            "persistent_price_registry_copied": True,
        },
    }
    (source / "snapshot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (source / "lineage" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _git(project: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
