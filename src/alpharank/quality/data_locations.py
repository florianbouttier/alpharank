"""Deterministic inventory of AlphaRank data locations and code readers."""

from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class DataLocationSpec:
    """One current data file or package and its intended destination."""

    location_id: str
    path: str
    kind: str
    current_role: str
    target_role: str
    migration_status: str


DATA_LOCATION_SPECS = (
    DataLocationSpec("data_root", "data", "package", "mixed_generation_root", "routing_only", "observed"),
    DataLocationSpec("legacy_prices", "data/US_Finalprice.parquet", "file", "legacy_model_input", "warehouse/raw/eodhd then mart", "compatibility"),
    DataLocationSpec("legacy_benchmark", "data/SP500Price.parquet", "file", "legacy_model_input", "warehouse/raw/eodhd then mart", "compatibility"),
    DataLocationSpec("legacy_constituents", "data/SP500_Constituents.csv", "file", "legacy_model_input", "warehouse/raw then mart", "compatibility"),
    DataLocationSpec("legacy_general", "data/US_General.parquet", "file", "legacy_model_input", "warehouse/def then mart", "compatibility"),
    DataLocationSpec("legacy_income", "data/US_Income_statement.parquet", "file", "legacy_model_input", "warehouse/def then mart", "compatibility"),
    DataLocationSpec("legacy_balance", "data/US_Balance_sheet.parquet", "file", "legacy_model_input", "warehouse/def then mart", "compatibility"),
    DataLocationSpec("legacy_cash_flow", "data/US_Cash_flow.parquet", "file", "legacy_model_input", "warehouse/def then mart", "compatibility"),
    DataLocationSpec("legacy_earnings", "data/US_Earnings.parquet", "file", "legacy_model_input", "warehouse/def then mart", "compatibility"),
    DataLocationSpec("legacy_shares", "data/US_share.parquet", "file", "legacy_model_input", "warehouse/def then mart", "compatibility"),
    DataLocationSpec("legacy_latest_pointer", "data/latest_snapshot.json", "file", "legacy_mutable_pointer", "model_inputs/manifests/latest.json", "compatibility"),
    DataLocationSpec("local_snapshots", "data/_snapshots", "package", "historical_local_snapshots", "archive_by_reference", "legacy"),
    DataLocationSpec("eodhd_archive", "data/eodhd", "package", "immutable_paid_source_archive", "warehouse/raw/eodhd", "catalogued"),
    DataLocationSpec("open_source_cache", "data/open_source/_cache", "package", "disposable_transport_cache", "outside_raw_contract", "retained"),
    DataLocationSpec("open_source_transactions", "data/open_source/_transactions", "package", "transaction_work_area", "ephemeral_transaction_state", "retained"),
    DataLocationSpec("open_source_archive", "data/open_source/archive", "package", "historical_research_packages", "archive_by_reference", "legacy"),
    DataLocationSpec("open_source_audit", "data/open_source/audit", "package", "data_quality_evidence", "run_or_research_evidence", "retained"),
    DataLocationSpec("open_source_history", "data/open_source/history", "package", "published_package_history", "snapshot_history", "legacy"),
    DataLocationSpec("open_source_official", "data/open_source/official", "package", "mixed_source_ingestion_store", "warehouse_layers", "transition"),
    DataLocationSpec("open_source_raw", "data/open_source/official/raw", "package", "normalized_provider_history", "warehouse/raw then stg", "transition"),
    DataLocationSpec("open_source_target", "data/open_source/official/target", "package", "mixed_selected_outputs", "warehouse/def or mart", "transition"),
    DataLocationSpec("open_source_manifests", "data/open_source/official/manifests", "package", "mutable_ingestion_pointers", "warehouse/manifests", "transition"),
    DataLocationSpec("open_source_runs", "data/open_source/official/runs", "package", "ingestion_run_evidence", "warehouse/raw receipts", "transition"),
    DataLocationSpec("open_source_output", "data/open_source/output", "package", "mixed_source_research_export", "non_production_replay_only", "legacy"),
    DataLocationSpec("sec_legacy", "data/sec", "package", "sec_only_generations_and_staging", "warehouse/raw stg def", "transition"),
    DataLocationSpec("warehouse", "data/warehouse", "package", "canonical_transformation_root", "warehouse", "target"),
    DataLocationSpec("warehouse_raw", "data/warehouse/raw", "package", "provider_observations_and_receipts", "warehouse/raw", "target"),
    DataLocationSpec("warehouse_stg", "data/warehouse/stg", "package", "source_neutral_normalization", "warehouse/stg", "target"),
    DataLocationSpec("warehouse_def", "data/warehouse/def", "package", "governed_value_selection", "warehouse/def", "target"),
    DataLocationSpec("warehouse_mart", "data/warehouse/mart", "package", "consumer_ready_model_inputs", "warehouse/mart", "target"),
    DataLocationSpec("warehouse_manifests", "data/warehouse/manifests", "package", "migration_and_promotion_evidence", "warehouse/manifests", "target"),
    DataLocationSpec("model_input_history", "data/model_inputs/history", "package", "immutable_model_input_releases", "snapshot", "canonical"),
    DataLocationSpec("model_input_manifests", "data/model_inputs/manifests", "package", "atomic_snapshot_pointers", "snapshot_pointer", "canonical"),
    DataLocationSpec("production_pointers", "data/production", "package", "small_production_control_pointers", "warehouse/manifests", "transition"),
    DataLocationSpec("legacy_data_outputs", "data/outputs", "package", "legacy_data_checkpoints", "archive_by_reference", "legacy"),
)


def build_data_location_inventory(
    root: Path,
    *,
    observed_at: str,
    observation_root: Path | None = None,
) -> dict[str, object]:
    """Build one reviewable snapshot without hashing or copying data payloads."""

    root = root.resolve()
    observed_root = (observation_root or root).resolve()
    references = discover_data_references(root)
    readers_by_location: dict[str, set[str]] = {
        spec.location_id: set() for spec in DATA_LOCATION_SPECS
    }
    reference_rows: list[dict[str, str]] = []
    for reference, readers in sorted(references.items()):
        spec = _owning_spec(reference)
        reference_rows.append(
            {
                "reference": reference,
                "location_id": spec.location_id,
                "reader_count": str(len(readers)),
            }
        )
        readers_by_location[spec.location_id].update(readers)

    active_python = _active_python_paths(root)
    for spec in DATA_LOCATION_SPECS:
        if spec.kind != "file":
            continue
        basename = Path(spec.path).name
        for reader in active_python:
            if basename in (root / reader).read_text(encoding="utf-8"):
                readers_by_location[spec.location_id].add(reader)

    locations = []
    for spec in DATA_LOCATION_SPECS:
        observed = _observe_path(observed_root / spec.path, spec.kind)
        locations.append(
            {
                **asdict(spec),
                **observed,
                "readers": sorted(readers_by_location[spec.location_id]),
            }
        )
    return {
        "schema_version": 1,
        "inventory_id": "alpharank_data_location_inventory_v1",
        "observed_at": observed_at,
        "scope": "current files and packages; metadata only; no payload copied or downloaded",
        "summary": {
            "location_count": len(locations),
            "file_location_count": sum(row["kind"] == "file" for row in locations),
            "package_location_count": sum(row["kind"] == "package" for row in locations),
            "existing_location_count": sum(bool(row["exists"]) for row in locations),
            "static_reference_count": len(reference_rows),
            "reader_edge_count": sum(len(row["readers"]) for row in locations),
        },
        "locations": locations,
        "static_references": reference_rows,
    }


def discover_data_references(root: Path) -> dict[str, list[str]]:
    """Resolve simple tracked Python path expressions that start under data/."""

    references: dict[str, set[str]] = {}
    for relative_path in _active_python_paths(root):
        tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
        symbols = _path_symbols(tree)
        for node in ast.walk(tree):
            parts = _path_parts(node, symbols)
            if "data" not in parts:
                continue
            start = parts.index("data")
            normalized = "/".join(parts[start:])
            if normalized == "data" or normalized.startswith("data/"):
                references.setdefault(normalized, set()).add(relative_path)
    return {path: sorted(readers) for path, readers in sorted(references.items())}


def validate_data_location_inventory(
    root: Path,
    inventory: Mapping[str, object],
) -> dict[str, object]:
    """Validate declarations and reader edges while keeping observed sizes frozen."""

    errors: list[str] = []
    rows = inventory.get("locations")
    if not isinstance(rows, list):
        return {"passed": False, "errors": ["locations must be a list"]}
    declared = [asdict(spec) for spec in DATA_LOCATION_SPECS]
    observed_declarations = [
        {key: row.get(key) for key in declared[0]}
        for row in rows
        if isinstance(row, dict)
    ]
    if observed_declarations != declared:
        errors.append("location declarations differ from the maintained registry")

    expected = build_data_location_inventory(
        root,
        observed_at=str(inventory.get("observed_at", "not_recorded")),
    )
    expected_locations = expected.get("locations")
    if not isinstance(expected_locations, list):
        return {"passed": False, "errors": ["generated locations must be a list"]}
    expected_rows = {
        str(row["location_id"]): row
        for row in expected_locations
        if isinstance(row, dict)
    }
    for row in rows:
        if not isinstance(row, dict):
            errors.append("location row must be an object")
            continue
        location_id = str(row.get("location_id"))
        if location_id not in expected_rows:
            continue
        if row.get("readers") != expected_rows[location_id]["readers"]:
            errors.append(f"reader drift for {location_id}")
        if not isinstance(row.get("file_count"), int) or int(row["file_count"]) < 0:
            errors.append(f"invalid file_count for {location_id}")
        if not isinstance(row.get("size_bytes"), int) or int(row["size_bytes"]) < 0:
            errors.append(f"invalid size_bytes for {location_id}")
    return {
        "schema_version": 1,
        "inventory_id": inventory.get("inventory_id"),
        "location_count": len(rows),
        "errors": errors,
        "passed": not errors,
    }


def write_data_location_inventory(path: Path, inventory: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _active_python_paths(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", "scripts/*.py", "scripts/**/*.py", "src/alpharank/*.py", "src/alpharank/**/*.py"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(
        path
        for path in completed.stdout.decode().split("\0")
        if path
        and not path.startswith(
            ("scripts/_archive/", "scripts/maintenance/", "scripts/quality/", "src/alpharank/quality/")
        )
        and path not in {"scripts/validate_documentation.py", "scripts/validate_markdown_links.py"}
    )


def _path_symbols(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    symbols: dict[str, tuple[str, ...]] = {}
    assignments = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))]
    for _ in range(4):
        for assignment in assignments:
            value = assignment.value
            if value is None:
                continue
            parts = _path_parts(value, symbols)
            if not parts:
                continue
            targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    symbols[target.id] = parts
    return symbols


def _path_parts(node: ast.AST, symbols: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = node.value.replace("\\", "/")
        return tuple(part for part in value.split("/") if part and part != ".")
    if isinstance(node, ast.Name):
        return symbols.get(node.id, ())
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return (*_path_parts(node.left, symbols), *_path_parts(node.right, symbols))
    if isinstance(node, ast.Subscript):
        return _path_parts(node.value, symbols)
    if isinstance(node, ast.Attribute):
        return _path_parts(node.value, symbols)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "Path" and node.args:
            return _path_parts(node.args[0], symbols)
        if isinstance(node.func, ast.Attribute):
            base = _path_parts(node.func.value, symbols)
            if node.func.attr == "joinpath":
                return (*base, *sum((_path_parts(arg, symbols) for arg in node.args), ()))
            if node.func.attr in {"resolve", "expanduser", "absolute"}:
                return base
    return ()


def _owning_spec(reference: str) -> DataLocationSpec:
    candidates = [
        spec
        for spec in DATA_LOCATION_SPECS
        if reference == spec.path or reference.startswith(f"{spec.path}/")
    ]
    if not candidates:
        return DATA_LOCATION_SPECS[0]
    return max(candidates, key=lambda spec: len(spec.path))


def _observe_path(path: Path, kind: str) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "file_count": 0, "size_bytes": 0}
    if kind == "file":
        return {"exists": path.is_file(), "file_count": int(path.is_file()), "size_bytes": path.stat().st_size if path.is_file() else 0}
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return {
        "exists": path.is_dir(),
        "file_count": len(files),
        "size_bytes": sum(candidate.stat().st_size for candidate in files),
    }
