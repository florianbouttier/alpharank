from __future__ import annotations

import filecmp
from pathlib import Path
import os
import shutil
from typing import Any

import polars as pl

from alpharank.data.output_history import snapshot_output_directory
from alpharank.data.open_source.storage import write_json


class PublishedOutputResult:
    def __init__(self, published_paths: dict[str, Path], snapshot_dir: Path | None) -> None:
        self.published_paths = published_paths
        self.snapshot_dir = snapshot_dir


def publish_open_source_output_package(
    *,
    output_dir: Path,
    legacy_paths: dict[str, Path],
    constituents_source_path: Path,
    prices_frame: pl.DataFrame,
    prices_lineage: pl.DataFrame | None = None,
    persistent_price_history_registry: pl.DataFrame | None = None,
    benchmark_prices: pl.DataFrame,
    general_reference: pl.DataFrame,
    general_reference_lineage: pl.DataFrame,
    consolidated_financials: pl.DataFrame,
    consolidated_lineage: pl.DataFrame,
    source_summary: pl.DataFrame,
    earnings_consolidated: pl.DataFrame,
    earnings_lineage: pl.DataFrame,
    earnings_long_frame: pl.DataFrame,
    manifest: dict[str, Any] | None = None,
    history_root: Path | None = None,
    snapshot_prefix: str = "open_source_output",
) -> PublishedOutputResult:
    _validate_source_refresh_contract(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    lineage_dir = output_dir / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)

    published: dict[str, Path] = {}
    allowed_output_files = set(legacy_paths) | {"SP500_Constituents.csv", "README.md"}
    for existing in output_dir.iterdir():
        if existing.name == "lineage" or existing.name in allowed_output_files:
            continue
        if existing.is_file():
            existing.unlink()

    for file_name, source_path in legacy_paths.items():
        destination = output_dir / file_name
        _copy_if_changed(source_path, destination)
        published[file_name] = destination

    constituents_destination = output_dir / "SP500_Constituents.csv"
    _copy_if_changed(constituents_source_path, constituents_destination)
    published["SP500_Constituents.csv"] = constituents_destination

    lineage_outputs = {
        "prices_open_source.parquet": prices_frame,
        "prices_open_source_lineage.parquet": prices_lineage if prices_lineage is not None else prices_frame,
        "benchmark_prices_open_source.parquet": benchmark_prices,
        "general_reference.parquet": general_reference,
        "general_reference_lineage.parquet": general_reference_lineage,
        "earnings_open_source_consolidated.parquet": earnings_consolidated,
        "earnings_open_source_lineage.parquet": earnings_lineage,
        "earnings_open_source_long.parquet": earnings_long_frame,
        "financials_open_source_consolidated.parquet": consolidated_financials,
        "financials_open_source_lineage.parquet": consolidated_lineage,
        "financials_open_source_source_summary.parquet": source_summary,
    }
    if persistent_price_history_registry is not None:
        lineage_outputs["persistent_price_history_registry.parquet"] = (
            persistent_price_history_registry
        )
    staging_lineage_paths: dict[str, Path] = {}
    legacy_parent_dirs = {path.parent for path in legacy_paths.values()}
    if len(legacy_parent_dirs) == 1:
        staging_lineage_dir = next(iter(legacy_parent_dirs)) / "lineage"
        if staging_lineage_dir.exists():
            staging_lineage_paths = {
                path.name: path
                for path in staging_lineage_dir.glob("*")
                if path.is_file()
            }

    allowed_lineage_files = set(lineage_outputs) | {"manifest.json"} | set(staging_lineage_paths)
    for existing in lineage_dir.iterdir():
        if existing.name in allowed_lineage_files:
            continue
        if existing.is_file():
            existing.unlink()
    for file_name, frame in lineage_outputs.items():
        path = lineage_dir / file_name
        _write_parquet_if_changed(frame, path)
        published[f"lineage/{file_name}"] = path
    for file_name, source_path in staging_lineage_paths.items():
        destination = lineage_dir / file_name
        _copy_if_changed(source_path, destination)
        published[f"lineage/{file_name}"] = destination

    if manifest is not None:
        manifest_path = lineage_dir / "manifest.json"
        write_json(manifest_path, manifest)
        published["lineage/manifest.json"] = manifest_path

    snapshot_dir = (
        snapshot_output_directory(
            output_dir,
            history_root=history_root,
            snapshot_prefix=snapshot_prefix,
            metadata=manifest,
        )
        if history_root is not None
        else None
    )

    return PublishedOutputResult(published_paths=published, snapshot_dir=snapshot_dir)


def _validate_source_refresh_contract(manifest: dict[str, Any] | None) -> None:
    if manifest is None:
        return
    contract = manifest.get("source_refresh_contract")
    if not isinstance(contract, dict):
        return
    if contract.get("snapshot_scope") != "full_ingestion":
        raise RuntimeError(
            "Only a guarded full_ingestion may replace the canonical open-source "
            "output package. Diagnostic refreshes must remain non-published."
        )
    policy = contract.get("policy")
    if not isinstance(policy, dict) or not policy.get("require_eodhd_price_seed"):
        raise RuntimeError("Production publication requires the immutable EODHD price seed")
    gate = contract.get("price_revision_guard")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise RuntimeError("Production publication requires a passed price revision guard")


def _copy_if_changed(source: Path, destination: Path) -> None:
    if destination.exists() and filecmp.cmp(source, destination, shallow=False):
        return
    temporary = destination.with_name(f".{destination.name}.publishing-tmp")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _write_parquet_if_changed(frame: pl.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.publishing-tmp")
    temporary.unlink(missing_ok=True)
    frame.write_parquet(temporary)
    if destination.exists() and filecmp.cmp(temporary, destination, shallow=False):
        temporary.unlink()
        return
    os.replace(temporary, destination)
