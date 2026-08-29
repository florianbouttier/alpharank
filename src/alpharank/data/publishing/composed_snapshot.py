"""Build immutable model inputs from independently validated data packages."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from alpharank.data.prices.history import PERSISTENT_PRICE_HISTORY_POLICY_ID
from alpharank.data.prices.ticker_transitions import PRICE_TICKER_TRANSITION_POLICY_ID
from alpharank.data.publishing.snapshot_storage import copy_snapshot_file
from alpharank.data.security_identity import (
    SECURITY_IDENTITY_POLICY_ID,
    assert_security_identity_compliance,
    assert_security_identity_reference_compliance,
    load_security_identity_registry,
)

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


@dataclass(frozen=True)
class SnapshotCompositionRequest:
    price_package_dir: Path
    sec_package_dir: Path
    price_manifest: Mapping[str, Any]
    sec_manifest: Mapping[str, Any]
    security_identity: Mapping[str, Any]
    price_ticker_transition: Mapping[str, Any]


@dataclass(frozen=True)
class SnapshotCompositionContext:
    sources: dict[str, dict[str, Any]]
    composition_id: str


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
    price_ticker_transition = _validate_price_ticker_transition_package(
        price_package_dir,
        price_manifest,
    )
    sec_manifest = _validate_sec_package(sec_package_dir)
    security_identity = _validate_security_identity_packages(
        price_package_dir=price_package_dir,
        price_manifest=price_manifest,
        sec_package_dir=sec_package_dir,
        sec_manifest=sec_manifest,
    )
    source_price_identity = _price_payload_identity(price_package_dir / "US_Finalprice.parquet")

    composition = _build_snapshot_composition(
        SnapshotCompositionRequest(
            price_package_dir=price_package_dir,
            sec_package_dir=sec_package_dir,
            price_manifest=price_manifest,
            sec_manifest=sec_manifest,
            security_identity=security_identity,
            price_ticker_transition=price_ticker_transition,
        )
    )
    sources = composition.sources
    composition_id = composition.composition_id
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
            storage_modes.append(copy_snapshot_file(price_package_dir / name, staging_dir / name))
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
        expected_hashes = {Path(key).name: record["sha256"] for key, record in sources.items()}
        if output_hashes != expected_hashes:
            raise RuntimeError("Composed snapshot copy verification failed")
        snapshot_price_identity = _price_payload_identity(staging_dir / "US_Finalprice.parquet")
        if snapshot_price_identity != source_price_identity:
            raise RuntimeError("Composed snapshot changed the canonical price payload")
        persistent_registry_copied = int(price_manifest.get("contract_version", 1)) >= 2
        if persistent_registry_copied:
            source_registry = (
                price_package_dir / "lineage" / "persistent_price_history_registry.parquet"
            )
            copied_registry = (
                staging_dir / "lineage" / "prices" / "persistent_price_history_registry.parquet"
            )
            if _sha256(copied_registry) != _sha256(source_registry):
                raise RuntimeError("Composed snapshot changed the price registry")

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
                    "manifest_sha256": _sha256(price_package_dir / "lineage" / "manifest.json"),
                },
                "sec": {
                    "path": str(sec_package_dir),
                    "run_id": sec_manifest.get("run_id"),
                    "manifest_sha256": _sha256(sec_package_dir / "lineage" / "manifest.json"),
                },
            },
            "source_files": sources,
            "output_sha256": output_hashes,
            "data_freshness": price_manifest.get("data_freshness", {}),
            "price_payload_identity": snapshot_price_identity,
            "security_identity": security_identity,
            "price_ticker_transition": price_ticker_transition,
            "validation": {
                "price_contract": (
                    "full_ingestion + preceding validated published lineage + "
                    "EODHD coverage + passed revision gate"
                ),
                "fundamental_contract": "strict SEC-only",
                "same_snapshot_for_legacy_and_boosting": True,
                "price_payload_preserved_exactly": True,
                "persistent_price_registry_copied": persistent_registry_copied,
                "security_identity_policy_applied": security_identity["policy_required"],
                "price_ticker_transition_policy_applied": price_ticker_transition[
                    "policy_required"
                ],
                "passed": True,
            },
            "storage": {
                "strategy": "copy_on_write_with_physical_copy_fallback",
                "file_count": len(storage_modes),
                "storage_mode_counts": {
                    mode: storage_modes.count(mode) for mode in sorted(set(storage_modes))
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
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _build_snapshot_composition(
    request: SnapshotCompositionRequest,
) -> SnapshotCompositionContext:
    sources = {
        **{f"price/{name}": _file_record(request.price_package_dir / name) for name in PRICE_FILES},
        **{f"sec/{name}": _file_record(request.sec_package_dir / name) for name in SEC_FILES},
        **{
            f"sec/{name}": _file_record(request.sec_package_dir / name)
            for name in OPTIONAL_SEC_FILES
            if (request.sec_package_dir / name).exists()
        },
    }
    payload = {
        "contract_version": 2,
        "source_files": {key: record["sha256"] for key, record in sorted(sources.items())},
        "price_manifest_sha256": _sha256(request.price_package_dir / "lineage" / "manifest.json"),
        "persistent_price_registry_sha256": (
            _sha256(
                request.price_package_dir / "lineage" / "persistent_price_history_registry.parquet"
            )
            if int(request.price_manifest.get("contract_version", 1)) >= 2
            else None
        ),
        "price_run_id": request.price_manifest.get("run_id"),
        "sec_run_id": request.sec_manifest.get("run_id"),
        "security_identity_registry_sha256": request.security_identity.get("registry_sha256"),
        "price_ticker_transition_registry_sha256": request.price_ticker_transition.get(
            "registry_sha256"
        ),
    }
    composition_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return SnapshotCompositionContext(sources=sources, composition_id=composition_id)


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
    observed = {str(name): _sha256(snapshot_dir / str(name)) for name in expected}
    if observed != dict(expected):
        differing = sorted(name for name in expected if observed.get(name) != expected.get(name))
        raise RuntimeError(f"Composed snapshot hash mismatch: {differing}")
    expected_identity = manifest.get("price_payload_identity")
    if expected_identity is not None:
        observed_identity = _price_payload_identity(snapshot_dir / "US_Finalprice.parquet")
        if observed_identity != expected_identity:
            raise RuntimeError("Composed snapshot price identity mismatch")
    if manifest.get("validation", {}).get("persistent_price_registry_copied"):
        registry = snapshot_dir / "lineage" / "prices" / "persistent_price_history_registry.parquet"
        if not registry.is_file():
            raise RuntimeError("Composed snapshot lost its persistent price registry")
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
    gate = contract.get("price_publication_guard", contract.get("price_revision_guard"))
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise RuntimeError("Price package did not pass the price publication gate")
    if int(manifest.get("contract_version", 1)) >= 2:
        persistent = contract.get("persistent_price_history")
        if (
            not isinstance(persistent, Mapping)
            or persistent.get("policy_id") != PERSISTENT_PRICE_HISTORY_POLICY_ID
            or persistent.get("routine_deletion_allowed") is not False
        ):
            raise RuntimeError("Price package has no valid persistent-history contract")
        validation = manifest.get("validation")
        if (
            not isinstance(validation, Mapping)
            or validation.get("all_previous_validated_inactive_history_preserved") is not True
            or validation.get("open_source_only_inactive_history_persisted") is not True
        ):
            raise RuntimeError("Price package did not prove persistent inactive history")
        registry = package_dir / "lineage" / "persistent_price_history_registry.parquet"
        if not registry.is_file():
            raise RuntimeError("Price package is missing its persistent-history registry")
        registry_record = manifest.get("artifacts", {}).get("persistent_price_history_registry", {})
        if not registry_record.get("sha256"):
            raise RuntimeError("Persistent-history registry has no manifest hash")
        if _sha256(registry) != registry_record["sha256"]:
            raise RuntimeError("Persistent-history registry hash mismatch")
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
    source_column = "selected_source" if "selected_source" in financial_schema else "source"
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


def _validate_price_ticker_transition_package(
    package_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    contract = manifest.get("source_refresh_contract", {}).get("price_ticker_transition")
    if contract is None:
        return {"policy_required": False, "registry_sha256": None, "passed": True}
    if not isinstance(contract, Mapping):
        raise RuntimeError("Price ticker transition contract is malformed")
    if contract.get("policy_id") != PRICE_TICKER_TRANSITION_POLICY_ID:
        raise RuntimeError("Price ticker transition policy id is invalid")
    if contract.get("passed") is not True or contract.get("manual_price_values") != 0:
        raise RuntimeError("Price ticker transition did not pass its additive-return contract")
    lineage_dir = package_dir / "lineage"
    paths = {
        "registry": lineage_dir / "price_ticker_transition_policy.json",
        "audit": lineage_dir / "price_ticker_transition_audit.parquet",
    }
    for key, path in paths.items():
        record = contract.get(key)
        if not path.is_file() or not isinstance(record, Mapping):
            raise RuntimeError(f"Price ticker transition package is missing {key}")
        if record.get("sha256") != _sha256(path):
            raise RuntimeError(f"Price ticker transition {key} hash mismatch")
    return {
        "policy_id": PRICE_TICKER_TRANSITION_POLICY_ID,
        "policy_required": True,
        "registry_sha256": _sha256(paths["registry"]),
        "audit_sha256": _sha256(paths["audit"]),
        "added_rows": contract.get("added_rows"),
        "passed": True,
    }


def _validate_security_identity_packages(
    *,
    price_package_dir: Path,
    price_manifest: Mapping[str, Any],
    sec_package_dir: Path,
    sec_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    registry = load_security_identity_registry()
    identity_roots = set(registry.get_column("canonical_ticker").to_list()) | set(
        registry.get_column("source_ticker").to_list()
    )
    dated_files = (
        (price_package_dir / "US_Finalprice.parquet", "ticker", "date"),
        (price_package_dir / "SP500_Constituents.csv", "Ticker", "Date"),
        (sec_package_dir / "US_Income_statement.parquet", "ticker", "date"),
        (sec_package_dir / "US_Balance_sheet.parquet", "ticker", "date"),
        (sec_package_dir / "US_Cash_flow.parquet", "ticker", "date"),
        (sec_package_dir / "US_Earnings.parquet", "ticker", "date"),
        (sec_package_dir / "US_share.parquet", "ticker", "date"),
    )
    checked_files: list[str] = []
    policy_required = False
    for path, ticker_column, date_column in dated_files:
        if not path.is_file():
            continue
        frame = (
            pl.read_csv(path, infer_schema_length=0)
            if path.suffix == ".csv"
            else pl.read_parquet(path)
        )
        if ticker_column not in frame.columns or date_column not in frame.columns:
            continue
        roots = set(
            frame.get_column(ticker_column)
            .drop_nulls()
            .cast(pl.String)
            .str.to_uppercase()
            .str.replace(r"\.US$", "")
            .unique()
            .to_list()
        )
        if not roots.intersection(identity_roots):
            continue
        policy_required = True
        assert_security_identity_compliance(
            frame,
            ticker_column=ticker_column,
            date_column=date_column,
            registry=registry,
        )
        checked_files.append(path.name)

    general_path = sec_package_dir / "US_General.parquet"
    if general_path.is_file():
        general = pl.read_parquet(general_path)
        if {"Code", "CIK"}.issubset(general.columns):
            roots = set(
                general.get_column("Code")
                .drop_nulls()
                .cast(pl.String)
                .str.to_uppercase()
                .str.replace(r"\.US$", "")
                .unique()
                .to_list()
            )
            if roots.intersection(identity_roots):
                policy_required = True
                assert_security_identity_reference_compliance(
                    general,
                    ticker_column="Code",
                    cik_columns=("CIK",),
                    registry=registry,
                )
                checked_files.append(general_path.name)

    if policy_required:
        price_policy = (
            price_manifest.get("source_refresh_contract", {})
            .get("security_identity", {})
            .get("policy_id")
        )
        sec_policy = sec_manifest.get("security_identity", {}).get("policy_id")
        if price_policy != SECURITY_IDENTITY_POLICY_ID:
            raise RuntimeError("Price package does not declare the security identity policy")
        if sec_policy != SECURITY_IDENTITY_POLICY_ID:
            raise RuntimeError("SEC package does not declare the security identity policy")
    registry_path = Path(registry.get_column("registry_path").drop_nulls().unique().item())
    return {
        "policy_id": SECURITY_IDENTITY_POLICY_ID,
        "policy_required": policy_required,
        "registry_path": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "checked_files": sorted(checked_files),
        "passed": True,
    }


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


def _price_payload_identity(path: Path) -> dict[str, Any]:
    frame = pl.read_parquet(path)
    ticker_column = "ticker" if "ticker" in frame.columns else "Ticker"
    date_column = "date" if "date" in frame.columns else "Date"
    missing = [column for column in (ticker_column, date_column) if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Canonical price payload has no key columns: {missing}")
    canonical = frame.sort(ticker_column, date_column).select(sorted(frame.columns))
    buffer = BytesIO()
    canonical.write_ipc(buffer, compression="uncompressed")
    unique_key_count = frame.select(pl.struct(ticker_column, date_column).n_unique()).item()
    return {
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "row_count": frame.height,
        "key_columns": [ticker_column, date_column],
        "unique_key_count": int(unique_key_count),
        "duplicate_key_count": int(frame.height - unique_key_count),
        "economic_series_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
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
