"""Build immutable model inputs from independently validated data packages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import polars as pl

from alpharank.data.snapshot_storage import copy_snapshot_file


PRICE_FILES = (
    "US_Finalprice.parquet",
    "SP500Price.parquet",
    "SP500_Constituents.csv",
)
SEC_FILES = (
    "US_General.parquet",
    "US_Income_statement.parquet",
    "US_Balance_sheet.parquet",
    "US_Cash_flow.parquet",
    "US_Earnings.parquet",
)
OPTIONAL_SEC_FILES = ("US_share.parquet",)
ALLOWED_FINANCIAL_SOURCES = frozenset({"sec_companyfacts", "sec_filing"})
ALLOWED_EARNINGS_SOURCES = frozenset(
    {
        "sec_companyfacts",
        "sec_filing",
        "sec_derived_eps",
        "sec_submissions",
    }
)


@dataclass(frozen=True)
class ComposedSnapshotResult:
    snapshot_dir: Path
    manifest_path: Path
    composition_id: str
    manifest: dict[str, Any]


def build_composed_model_snapshot(
    *,
    price_package_dir: Path,
    sec_package_dir: Path,
    history_root: Path,
    latest_manifest_path: Path | None = None,
    expected_through: str | None = None,
) -> ComposedSnapshotResult:
    """Compose one immutable, hash-addressed Legacy/Boosting input package."""

    price_package_dir = price_package_dir.resolve()
    sec_package_dir = sec_package_dir.resolve()
    history_root = history_root.resolve()
    price_manifest = _validate_price_package(
        price_package_dir,
        expected_through=expected_through,
    )
    sec_manifest = _validate_sec_package(sec_package_dir)

    sources = {
        **{
            f"price/{name}": _file_record(price_package_dir / name)
            for name in PRICE_FILES
        },
        **{
            f"sec/{name}": _file_record(sec_package_dir / name)
            for name in SEC_FILES
        },
        **{
            f"sec/{name}": _file_record(sec_package_dir / name)
            for name in OPTIONAL_SEC_FILES
            if (sec_package_dir / name).exists()
        },
    }
    composition_payload = {
        "contract_version": 1,
        "source_files": {
            key: record["sha256"] for key, record in sorted(sources.items())
        },
        "price_run_id": price_manifest.get("run_id"),
        "sec_run_id": sec_manifest.get("run_id"),
    }
    composition_id = hashlib.sha256(
        json.dumps(composition_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    generated_at = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"alpharank_input_{timestamp}_{composition_id[:12]}"
    history_root.mkdir(parents=True, exist_ok=True)
    snapshot_dir = history_root / snapshot_name
    if snapshot_dir.exists():
        raise FileExistsError(f"Composed snapshot already exists: {snapshot_dir}")
    staging_dir = history_root / f".{snapshot_name}.staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    storage_modes: list[str] = []
    try:
        for name in PRICE_FILES:
            storage_modes.append(
                copy_snapshot_file(price_package_dir / name, staging_dir / name)
            )
        for name in (*SEC_FILES, *OPTIONAL_SEC_FILES):
            source = sec_package_dir / name
            if source.exists():
                storage_modes.append(copy_snapshot_file(source, staging_dir / name))

        _copy_tree_if_present(
            price_package_dir / "lineage",
            staging_dir / "lineage" / "prices",
            storage_modes,
        )
        _copy_tree_if_present(
            sec_package_dir / "lineage",
            staging_dir / "lineage" / "sec",
            storage_modes,
        )

        output_hashes = {
            name: _sha256(staging_dir / name)
            for name in (*PRICE_FILES, *SEC_FILES, *OPTIONAL_SEC_FILES)
            if (staging_dir / name).exists()
        }
        expected_hashes = {
            Path(key).name: record["sha256"] for key, record in sources.items()
        }
        if output_hashes != expected_hashes:
            raise RuntimeError("Composed snapshot copy verification failed")

        manifest: dict[str, Any] = {
            "contract_version": 1,
            "scope": "alpharank_composed_model_input",
            "composition_id": composition_id,
            "generated_at": generated_at,
            "snapshot_dir": str(snapshot_dir),
            "source_packages": {
                "prices": {
                    "path": str(price_package_dir),
                    "run_id": price_manifest.get("run_id"),
                    "manifest_sha256": _sha256(
                        price_package_dir / "lineage" / "manifest.json"
                    ),
                },
                "sec": {
                    "path": str(sec_package_dir),
                    "run_id": sec_manifest.get("run_id"),
                    "manifest_sha256": _sha256(
                        sec_package_dir / "lineage" / "manifest.json"
                    ),
                },
            },
            "source_files": sources,
            "output_sha256": output_hashes,
            "data_freshness": price_manifest.get("data_freshness", {}),
            "validation": {
                "price_contract": "full_ingestion + EODHD seed + passed revision gate",
                "fundamental_contract": "strict SEC-only",
                "same_snapshot_for_legacy_and_boosting": True,
                "passed": True,
            },
            "storage": {
                "strategy": "copy_on_write_with_physical_copy_fallback",
                "file_count": len(storage_modes),
                "storage_mode_counts": {
                    mode: storage_modes.count(mode)
                    for mode in sorted(set(storage_modes))
                },
            },
        }
        manifest_path = staging_dir / "lineage" / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_json(staging_dir / "snapshot_manifest.json", manifest)
        os.replace(staging_dir, snapshot_dir)

        final_manifest_path = snapshot_dir / "lineage" / "manifest.json"
        if latest_manifest_path is not None:
            latest_payload = {
                "composition_id": composition_id,
                "snapshot_dir": str(snapshot_dir),
                "manifest_path": str(final_manifest_path),
                "generated_at": generated_at,
            }
            _write_json_atomic(latest_manifest_path.resolve(), latest_payload)
        return ComposedSnapshotResult(
            snapshot_dir=snapshot_dir,
            manifest_path=final_manifest_path,
            composition_id=composition_id,
            manifest=manifest,
        )
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def validate_composed_model_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    """Revalidate hashes and source contracts of an existing composed snapshot."""

    snapshot_dir = snapshot_dir.resolve()
    manifest_path = snapshot_dir / "lineage" / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("scope") != "alpharank_composed_model_input":
        raise RuntimeError("Not an AlphaRank composed model snapshot")
    if manifest.get("validation", {}).get("passed") is not True:
        raise RuntimeError("Composed snapshot manifest is not marked valid")
    expected = manifest.get("output_sha256")
    if not isinstance(expected, Mapping):
        raise RuntimeError("Composed snapshot manifest has no output hashes")
    observed = {
        str(name): _sha256(snapshot_dir / str(name)) for name in expected
    }
    if observed != dict(expected):
        differing = sorted(
            name for name in expected if observed.get(name) != expected.get(name)
        )
        raise RuntimeError(f"Composed snapshot hash mismatch: {differing}")
    return {
        "composition_id": manifest["composition_id"],
        "snapshot_dir": str(snapshot_dir),
        "file_count": len(observed),
        "passed": True,
    }


def _validate_price_package(
    package_dir: Path,
    *,
    expected_through: str | None,
) -> dict[str, Any]:
    _require_files(package_dir, PRICE_FILES)
    manifest = _read_json(package_dir / "lineage" / "manifest.json")
    contract = manifest.get("source_refresh_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("Price package has no source refresh contract")
    if contract.get("snapshot_scope") != "full_ingestion":
        raise RuntimeError("Price package is not a full ingestion")
    policy = contract.get("policy")
    if not isinstance(policy, Mapping) or policy.get("require_eodhd_price_seed") is not True:
        raise RuntimeError("Price package does not require the immutable EODHD seed")
    seed = contract.get("eodhd_price_seed")
    if not isinstance(seed, Mapping) or not seed.get("sha256"):
        raise RuntimeError("Price package does not record the EODHD seed hash")
    gate = contract.get("price_revision_guard")
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise RuntimeError("Price package did not pass the price revision gate")
    if expected_through is not None:
        freshness = manifest.get("data_freshness", {})
        market_date = freshness.get("prices", {}).get("max_market_date")
        if market_date is None:
            raise RuntimeError(
                f"Invalid market freshness date: {market_date} for {expected_through}"
            )
        expected_date = date.fromisoformat(expected_through)
        observed_date = date.fromisoformat(str(market_date))
        if observed_date > expected_date or observed_date < expected_date - timedelta(days=7):
            raise RuntimeError(
                f"Invalid market freshness date: {market_date} for {expected_through}"
            )
    return manifest


def _validate_sec_package(package_dir: Path) -> dict[str, Any]:
    _require_files(package_dir, SEC_FILES)
    manifest = _read_json(package_dir / "lineage" / "manifest.json")
    if not str(manifest.get("scope", "")).startswith("sec_only"):
        raise RuntimeError("Fundamental package is not strict SEC-only")

    financial_path = package_dir / "lineage" / "financials_sec_lineage.parquet"
    earnings_path = package_dir / "lineage" / "earnings_sec_lineage.parquet"
    if not financial_path.exists() or not earnings_path.exists():
        raise RuntimeError("SEC package is missing value-level lineage")
    financials = pl.scan_parquet(financial_path)
    financial_schema = financials.collect_schema()
    source_column = (
        "selected_source" if "selected_source" in financial_schema else "source"
    )
    invalid_financial = (
        financials.filter(
            pl.col(source_column).is_not_null()
            & ~pl.col(source_column).is_in(sorted(ALLOWED_FINANCIAL_SOURCES))
        )
        .select(pl.col(source_column).unique())
        .collect()
    )
    if not invalid_financial.is_empty():
        raise RuntimeError(
            "SEC financial package contains forbidden sources: "
            f"{invalid_financial.get_column(source_column).to_list()}"
        )

    earnings = pl.scan_parquet(earnings_path)
    earnings_schema = earnings.collect_schema()
    for column in ("calendar_source", "actual_source"):
        if column not in earnings_schema:
            continue
        invalid = (
            earnings.filter(
                pl.col(column).is_not_null()
                & ~pl.col(column).is_in(sorted(ALLOWED_EARNINGS_SOURCES))
            )
            .select(pl.col(column).unique())
            .collect()
        )
        if not invalid.is_empty():
            raise RuntimeError(
                f"SEC earnings package contains forbidden {column}: "
                f"{invalid.get_column(column).to_list()}"
            )
    return manifest


def _require_files(directory: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Package {directory} is missing files: {missing}")


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tree_if_present(
    source: Path,
    destination: Path,
    storage_modes: list[str],
) -> None:
    if not source.exists():
        return
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        target = destination / path.relative_to(source)
        storage_modes.append(copy_snapshot_file(path, target))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Manifest must be a JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _write_json(temporary, payload)
    os.replace(temporary, path)
