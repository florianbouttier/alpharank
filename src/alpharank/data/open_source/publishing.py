from __future__ import annotations

from pathlib import Path
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
    benchmark_prices: pl.DataFrame,
    general_reference: pl.DataFrame,
    consolidated_financials: pl.DataFrame,
    consolidated_lineage: pl.DataFrame,
    source_summary: pl.DataFrame,
    earnings_frame: pl.DataFrame,
    earnings_long_frame: pl.DataFrame,
    manifest: dict[str, Any] | None = None,
    history_root: Path | None = None,
    snapshot_prefix: str = "open_source_output",
) -> PublishedOutputResult:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    lineage_dir = output_dir / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)

    published: dict[str, Path] = {}
    for file_name, source_path in legacy_paths.items():
        destination = output_dir / file_name
        shutil.copy2(source_path, destination)
        published[file_name] = destination

    constituents_destination = output_dir / "SP500_Constituents.csv"
    shutil.copy2(constituents_source_path, constituents_destination)
    published["SP500_Constituents.csv"] = constituents_destination

    lineage_outputs = {
        "prices_open_source.parquet": prices_frame,
        "benchmark_prices_open_source.parquet": benchmark_prices,
        "general_reference.parquet": general_reference,
        "earnings_open_source.parquet": earnings_frame,
        "earnings_open_source_long.parquet": earnings_long_frame,
        "financials_open_source_consolidated.parquet": consolidated_financials,
        "financials_open_source_lineage.parquet": consolidated_lineage,
        "financials_open_source_source_summary.parquet": source_summary,
    }
    for file_name, frame in lineage_outputs.items():
        path = lineage_dir / file_name
        frame.write_parquet(path)
        published[f"lineage/{file_name}"] = path

    if manifest is not None:
        manifest_path = lineage_dir / "manifest.json"
        write_json(manifest_path, manifest)
        published["lineage/manifest.json"] = manifest_path

    return PublishedOutputResult(published_paths=published, snapshot_dir=snapshot_dir)
