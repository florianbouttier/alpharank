from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from alpharank.data.publishing.promotion_evidence import seal_refresh_replay_inputs
from alpharank.data.publishing.snapshot_publication import validate_snapshot_publication
from alpharank.data.warehouse.mart import resolve_mart_model_input
from alpharank.data.warehouse.migration import (
    catalog_existing_eodhd,
    migrate_validated_snapshot_to_warehouse,
    promote_mart_pointer,
    rollback_mart_pointer,
    validate_eodhd_catalog,
    validate_warehouse_mart,
)

MODEL_FILES = (
    "US_Finalprice.parquet",
    "SP500Price.parquet",
    "SP500_Constituents.csv",
    "US_General.parquet",
    "US_Income_statement.parquet",
    "US_Balance_sheet.parquet",
    "US_Cash_flow.parquet",
    "US_Earnings.parquet",
    "US_share.parquet",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_composed_snapshot(
    root: Path, composition_id: str = "composition-test"
) -> tuple[Path, Path]:
    snapshot = root / "model_inputs" / "history" / "snapshot_old"
    snapshot.mkdir(parents=True)
    prices = pl.DataFrame(
        {
            "date": ["2026-07-31", "2026-07-31"],
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.0, 2.0],
            "volume": [100.0, 200.0],
            "adjusted_close": [1.0, 2.0],
            "ticker": ["AAA.US", "BBB.US"],
        }
    )
    prices.write_parquet(snapshot / "US_Finalprice.parquet")
    prices.select("ticker", "date", "adjusted_close", "close", "open", "high", "low", "volume").write_parquet(
        snapshot / "SP500Price.parquet"
    )
    (snapshot / "SP500_Constituents.csv").write_text("date,ticker\n2026-07-01,AAA.US\n")
    for name in MODEL_FILES[3:]:
        pl.DataFrame({"ticker": ["AAA.US"], "value": [1.0]}).write_parquet(snapshot / name)
    lineage = prices.with_columns(
        pl.lit("eodhd_frozen_history").alias("source"),
        pl.lit("US_Finalprice").alias("dataset"),
        pl.lit("seed").alias("ingestion_run_id"),
        pl.lit("2026-08-16T00:00:00+00:00").alias("ingested_at"),
    )
    lineage_dir = snapshot / "lineage" / "prices"
    lineage_dir.mkdir(parents=True)
    lineage.write_parquet(lineage_dir / "prices_open_source_lineage.parquet")
    (lineage_dir / "manifest.json").write_text("{}")
    output_hashes = {name: _sha256(snapshot / name) for name in MODEL_FILES}
    manifest = {
        "contract_version": 1,
        "scope": "alpharank_composed_model_input",
        "composition_id": composition_id,
        "snapshot_dir": str(snapshot),
        "output_sha256": output_hashes,
        "validation": {"passed": True},
    }
    (snapshot / "lineage" / "manifest.json").write_text(json.dumps(manifest))
    (snapshot / "snapshot_manifest.json").write_text(json.dumps(manifest))
    pointer = root / "model_inputs" / "manifests" / "latest.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(
        json.dumps(
            {
                "composition_id": composition_id,
                "snapshot_dir": str(snapshot),
                "manifest_path": str(snapshot / "lineage" / "manifest.json"),
                "generated_at": "2026-08-16T00:00:00+00:00",
            }
        )
    )
    return snapshot, pointer


def test_eodhd_catalog_deduplicates_bytes_and_reconstructs_every_source(tmp_path: Path) -> None:
    eodhd = tmp_path / "data" / "eodhd"
    first = eodhd / "history" / "run1" / "US_Finalprice.parquet"
    second = eodhd / "output" / "US_Finalprice.parquet"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"same immutable evidence")
    second.write_bytes(first.read_bytes())

    catalog_path = catalog_existing_eodhd(
        eodhd_root=eodhd,
        archive_dir=tmp_path / "warehouse" / "raw" / "eodhd",
        generated_at="2026-08-19T20:00:00+00:00",
    )
    validation = validate_eodhd_catalog(catalog_path)
    repeated = catalog_existing_eodhd(
        eodhd_root=eodhd,
        archive_dir=tmp_path / "warehouse" / "raw" / "eodhd",
        generated_at="2026-08-19T20:00:00+00:00",
    )

    assert repeated == catalog_path
    assert validation == {"passed": True, "source_file_count": 2, "unique_object_count": 1}


def test_live008_builds_byte_identical_mart_and_preserves_source(tmp_path: Path) -> None:
    snapshot, pointer = _seed_composed_snapshot(tmp_path / "data")
    eodhd = tmp_path / "data" / "eodhd" / "output"
    eodhd.mkdir(parents=True)
    (eodhd / "US_Finalprice.parquet").write_bytes(b"paid seed")
    source_inventory = {path.relative_to(snapshot): _sha256(path) for path in snapshot.rglob("*") if path.is_file()}

    result = migrate_validated_snapshot_to_warehouse(
        project_root=tmp_path,
        latest_pointer_path=pointer,
        warehouse_root=tmp_path / "data" / "warehouse",
        generated_at="2026-08-19T20:00:00+00:00",
    )
    validation = validate_warehouse_mart(
        mart_dir=result.mart_dir,
        def_manifest_path=result.def_manifest,
        source_snapshot_dir=snapshot,
    )

    assert result.model_file_count == 9
    assert validation["source_to_mart_hash_parity"] is True
    assert {name: _sha256(result.mart_dir / name) for name in MODEL_FILES} == {
        name: _sha256(snapshot / name) for name in MODEL_FILES
    }
    assert source_inventory == {
        path.relative_to(snapshot): _sha256(path) for path in snapshot.rglob("*") if path.is_file()
    }
    assert json.loads(pointer.read_text())["snapshot_dir"] == str(snapshot)


def test_mart_pointer_promotion_is_atomic_and_rollback_restores_exact_bytes(tmp_path: Path) -> None:
    snapshot, pointer = _seed_composed_snapshot(tmp_path / "data")
    eodhd = tmp_path / "data" / "eodhd" / "output"
    eodhd.mkdir(parents=True)
    (eodhd / "seed.json").write_text('{"source":"eodhd"}')
    result = migrate_validated_snapshot_to_warehouse(
        project_root=tmp_path,
        latest_pointer_path=pointer,
        warehouse_root=tmp_path / "data" / "warehouse",
        generated_at="2026-08-19T20:00:00+00:00",
    )
    before = pointer.read_bytes()

    promotion = promote_mart_pointer(
        pointer_path=pointer,
        mart_dir=result.mart_dir,
        promotion_root=tmp_path / "data" / "warehouse" / "manifests" / "promotions",
        migration_id=result.migration_id,
        generated_at="2026-08-19T20:01:00+00:00",
    )
    published = json.loads(pointer.read_text())
    publication_validation = validate_snapshot_publication(pointer)
    assert published["snapshot_dir"] == str(result.mart_dir)
    assert publication_validation["payload_copy_count"] == 0
    assert publication_validation["file_count"] == len(
        [path for path in result.mart_dir.rglob("*") if path.is_file()]
    )
    publication_dir = Path(published["publication_manifest_path"]).parent
    assert [path.name for path in publication_dir.iterdir()] == ["manifest.json"]

    rollback_mart_pointer(pointer_path=pointer, promotion_manifest_path=promotion)
    assert pointer.read_bytes() == before


def test_legacy_model_input_resolves_only_validated_mart_with_exact_parity(
    tmp_path: Path,
) -> None:
    snapshot, pointer = _seed_composed_snapshot(tmp_path / "data")
    eodhd = tmp_path / "data" / "eodhd" / "output"
    eodhd.mkdir(parents=True)
    (eodhd / "seed.json").write_text('{"source":"eodhd"}')
    source_hashes = {name: _sha256(snapshot / name) for name in MODEL_FILES}
    migrated = migrate_validated_snapshot_to_warehouse(
        project_root=tmp_path,
        latest_pointer_path=pointer,
        warehouse_root=tmp_path / "data" / "warehouse",
        promote=True,
        generated_at="2026-08-20T12:00:00+00:00",
    )

    resolved = resolve_mart_model_input(
        pointer,
        warehouse_root=tmp_path / "data" / "warehouse",
    )

    assert resolved.mart_dir == migrated.mart_dir
    assert resolved.composition_id == "composition-test"
    assert resolved.model_file_sha256 == source_hashes
    assert resolved.to_manifest()["source_snapshot_dir"] == str(snapshot.resolve())


def test_mart_switch_requires_replay_evidence_for_a_changed_composition(tmp_path: Path) -> None:
    baseline, pointer = _seed_composed_snapshot(tmp_path / "data")
    candidate_snapshot, candidate_pointer = _seed_composed_snapshot(
        tmp_path / "candidate", "candidate-test"
    )
    warehouse = tmp_path / "data" / "warehouse"
    seed = tmp_path / "data" / "eodhd" / "seed.json"
    seed.parent.mkdir()
    seed.write_text('{"source":"eodhd"}')
    candidate = migrate_validated_snapshot_to_warehouse(
        project_root=tmp_path,
        latest_pointer_path=candidate_pointer,
        warehouse_root=warehouse,
        generated_at="2026-08-20T12:00:00+00:00",
    )
    before = pointer.read_bytes()
    promotion_options = {
        "pointer_path": pointer,
        "mart_dir": candidate.mart_dir,
        "promotion_root": warehouse / "manifests" / "promotions",
        "migration_id": candidate.migration_id,
        "generated_at": "2026-08-20T12:01:00+00:00",
    }

    with pytest.raises(RuntimeError, match="requires sealed refresh replay evidence"):
        promote_mart_pointer(**promotion_options)
    assert pointer.read_bytes() == before

    report_path = _seed_refresh_replay_report(tmp_path, baseline, candidate_snapshot)
    report = json.loads(report_path.read_text())
    common_output = (
        Path(report["inputs"]["candidate_common"]) / "comparison_common_monthly.parquet"
    )
    common_output.write_bytes(b"changed after the audit")
    with pytest.raises(RuntimeError, match="changed artifact hashes"):
        promote_mart_pointer(
            **(promotion_options | {"generated_at": "2026-08-20T12:02:00+00:00"}),
            replay_report_path=report_path,
        )
    assert pointer.read_bytes() == before

    common_output.write_bytes(b"sealed replay table")
    report["historical_cutoff"] = "2026-06-01"
    report_path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="does not cover the published calendar"):
        promote_mart_pointer(
            **(promotion_options | {"generated_at": "2026-08-20T12:03:00+00:00"}),
            replay_report_path=report_path,
        )
    assert pointer.read_bytes() == before

    report["historical_cutoff"] = "2026-07-01"
    report_path.write_text(json.dumps(report))
    promote_mart_pointer(
        **(promotion_options | {"generated_at": "2026-08-20T12:04:00+00:00"}),
        replay_report_path=report_path,
    )
    published = json.loads(pointer.read_text())
    receipt = json.loads(Path(published["publication_manifest_path"]).read_text())
    assert published["composition_id"] == "candidate-test"
    assert receipt["promotion_evidence"]["sha256"] == _sha256(report_path)
    assert validate_snapshot_publication(pointer)["passed"] is True


def _seed_refresh_replay_report(root: Path, baseline: Path, candidate: Path) -> Path:
    dataset_map = {
        "final_price": "US_Finalprice.parquet",
        "sp500_price": "SP500Price.parquet",
        "sp500_constituents": "SP500_Constituents.csv",
        "income_statement": "US_Income_statement.parquet",
        "balance_sheet": "US_Balance_sheet.parquet",
        "cash_flow": "US_Cash_flow.parquet",
        "earnings": "US_Earnings.parquet",
    }
    inputs = {}
    for side, snapshot in (("baseline", baseline), ("candidate", candidate)):
        legacy_dir, boosting_dir, common_dir = (
            root / f"{side}_{role}" for role in ("legacy", "boosting", "common")
        )
        datasets = {key: {"sha256": _sha256(snapshot / name)} for key, name in dataset_map.items()}
        legacy_path = legacy_dir / "data_input_manifest.json"
        boosting_path = boosting_dir / "manifest.json"
        common_path = common_dir / "manifest.json"
        for key, path in (
            ("snapshot", snapshot), ("legacy", legacy_dir),
            ("boosting", boosting_dir), ("common", common_dir),
        ):
            inputs[f"{side}_{key}"] = str(path)
        for path in (
            legacy_dir / "legacy_common_holdings.parquet",
            legacy_dir / "legacy_common_monthly.parquet",
            legacy_dir / "legacy_detailed_returns_polars.parquet",
            boosting_dir / "classification_h06/predictions.parquet",
            common_dir / "comparison_common_holdings.parquet",
            common_dir / "comparison_common_monthly.parquet",
            common_dir / "boosting_live_score_holdings.parquet",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"sealed replay table")
        legacy_path.write_text(json.dumps({"datasets": datasets}))
        boosting_path.write_text(json.dumps({"input_paths": datasets}))
        common_path.write_text(json.dumps({
            "publication_eligible": True,
            "calendar": {"end_holding_month": "2026-07-01"},
            "lineage_check": {
                "passed": True,
                "left_manifest": str(legacy_path),
                "right_manifest": str(boosting_path),
            },
        }))
    report_path = root / "refresh_replay_report.json"
    report_path.write_text(json.dumps({
        "contract_version": 2,
        "status": "identical_historical_portfolios",
        "promotion_allowed_by_this_gate": True,
        "historical_cutoff": "2026-07-01",
        "vintage_portfolio_stability": {
            "status": "passed",
            "exact_match": True,
            "date_source": "baseline_live_portfolio",
            "decision_month": "2026-07-01",
        },
        "inputs": inputs,
        "promotion_input_sha256": seal_refresh_replay_inputs(inputs),
    }))
    return report_path


def test_mart_input_resolution_rejects_changed_model_bytes(tmp_path: Path) -> None:
    _, pointer = _seed_composed_snapshot(tmp_path / "data")
    eodhd = tmp_path / "data" / "eodhd" / "output"
    eodhd.mkdir(parents=True)
    (eodhd / "seed.json").write_text('{"source":"eodhd"}')
    migrated = migrate_validated_snapshot_to_warehouse(
        project_root=tmp_path,
        latest_pointer_path=pointer,
        warehouse_root=tmp_path / "data" / "warehouse",
        promote=True,
        generated_at="2026-08-20T12:00:00+00:00",
    )
    (migrated.mart_dir / "US_Finalprice.parquet").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="hash mismatch|bytes differ"):
        resolve_mart_model_input(
            pointer,
            warehouse_root=tmp_path / "data" / "warehouse",
        )


def test_snapshot_publication_detects_any_later_mart_file(tmp_path: Path) -> None:
    _, pointer = _seed_composed_snapshot(tmp_path / "data")
    eodhd = tmp_path / "data" / "eodhd" / "output"
    eodhd.mkdir(parents=True)
    (eodhd / "seed.json").write_text('{"source":"eodhd"}')
    migrated = migrate_validated_snapshot_to_warehouse(
        project_root=tmp_path,
        latest_pointer_path=pointer,
        warehouse_root=tmp_path / "data" / "warehouse",
        promote=True,
        generated_at="2026-08-20T12:00:00+00:00",
    )
    (migrated.mart_dir / "unexpected.txt").write_text("late mutation")

    with pytest.raises(RuntimeError, match="file inventory differs"):
        validate_snapshot_publication(pointer)
