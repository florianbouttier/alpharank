"""Verified sector-replay inputs for the standard performance report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from alpharank.data.warehouse.sector_sic_history import load_sic_sector_taxonomy
from alpharank.governance_contracts.common import sha256_path
from alpharank.replay.sector_allocation import SECTOR_CHAMPION_POLICY_ID, SECTOR_CHAMPION_TOP_N
from alpharank.replay.sector_backtest import REFERENCE_STRATEGIES, SECTOR_NATIVE, SECTOR_TREND


@dataclass(frozen=True, slots=True)
class SectorReportInputs:
    """Explicit REPLAY-014-format package, DEF release and manifest path root."""

    backtest_dir: Path
    sector_history_dir: Path
    repository_root: Path


@dataclass(frozen=True, slots=True)
class SectorReportData:
    """Sealed realized series and separately archived score-only positions."""

    monthly: pl.DataFrame
    holdings: pl.DataFrame
    pending: pl.DataFrame
    release_id: str
    taxonomy_id: str
    allowed_sectors: tuple[str, ...]
    source_paths: dict[str, Path]


def load_sector_report_data(
    inputs: SectorReportInputs,
    *,
    common_dir: Path,
    common_manifest: dict[str, Any],
    reference_monthly: pl.DataFrame,
) -> SectorReportData:
    """Read the sealed sector comparison and attach only the matching pending tail."""

    paths = {
        "sector_backtest_manifest": inputs.backtest_dir / "manifest.json",
        "sector_backtest_monthly": inputs.backtest_dir / "monthly_returns.parquet",
        "sector_backtest_holdings": inputs.backtest_dir / "sector_holdings.parquet",
        "sector_release_manifest": inputs.sector_history_dir / "manifest.json",
        "sector_history": inputs.sector_history_dir / "sector_history.parquet",
        "sector_taxonomy": inputs.repository_root
        / "configs/data_contracts/sec_sic_sector_taxonomy_v1.json",
    }
    backtest = _read_manifest(paths["sector_backtest_manifest"])
    _validate_backtest_reference(backtest, common_dir=common_dir, common_manifest=common_manifest)
    release = _read_manifest(paths["sector_release_manifest"])
    taxonomy = load_sic_sector_taxonomy(paths["sector_taxonomy"])
    sector_digest = release["files"]["sector_history.parquet"]["sha256"]
    _verify_sha(paths["sector_history"], sector_digest)
    for role in ("sector_backtest_monthly", "sector_backtest_holdings"):
        path = paths[role]
        _verify_sha(path, backtest["files"][path.name])
    monthly = pl.read_parquet(paths["sector_backtest_monthly"])
    holdings = pl.read_parquet(paths["sector_backtest_holdings"])
    _validate_reference_rows(monthly, reference_monthly)
    end = monthly["holding_month"].max()
    if not isinstance(end, date):
        raise ValueError("Sector backtest has no realized holding month.")
    pending, allocation_paths = _load_pending_allocations(
        inputs.repository_root,
        backtest=backtest,
        common_manifest=common_manifest,
        sector_digest=sector_digest,
        realized_end=end,
    )
    return SectorReportData(
        monthly,
        holdings,
        pending,
        release["release_id"],
        taxonomy.taxonomy_id,
        tuple(sorted(taxonomy.allowed_sectors)),
        paths | allocation_paths,
    )


def _validate_backtest_reference(
    backtest: dict[str, Any], *, common_dir: Path, common_manifest: dict[str, Any]
) -> None:
    if (
        backtest["contract_version"] != 1
        or backtest["status"] != "validated_r_and_d"
        or backtest["promotion_eligible"] is not False
    ):
        raise ValueError("Sector report requires a version-1 research backtest without promotion.")
    common_checks = (
        ("comparison_common_monthly.parquet", "inputs"),
        ("manifest.json", "input_manifests"),
    )
    for filename, key in common_checks:
        if sha256_path(common_dir / filename) not in backtest[key].values():
            raise ValueError(f"Sector backtest did not use the supplied common replay: {filename}")
    for role, filename in (
        ("holdings", "comparison_common_holdings.parquet"),
        ("boosting_live_score_holdings", "boosting_live_score_holdings.parquet"),
    ):
        _verify_sha(common_dir / filename, common_manifest["artifacts"][role]["sha256"])
    backtest_bps = backtest["methodology"]["transaction_cost_bps_times_turnover"]
    if backtest_bps != common_manifest["transaction_cost_policy"]["bps_times_turnover"]:
        raise ValueError("Sector backtest and common replay cost policies differ.")


def _validate_reference_rows(monthly: pl.DataFrame, reference_monthly: pl.DataFrame) -> None:
    keys = ["strategy", "holding_month"]
    expected = reference_monthly.filter(pl.col("strategy").is_in(REFERENCE_STRATEGIES)).sort(keys)
    actual = monthly.filter(pl.col("strategy").is_in(REFERENCE_STRATEGIES)).sort(keys)
    if not actual.select(expected.columns).equals(expected):
        raise ValueError("Sector backtest reference rows differ from the sealed common replay.")


def _load_pending_allocations(
    repository_root: Path,
    *,
    backtest: dict[str, Any],
    common_manifest: dict[str, Any],
    sector_digest: str,
    realized_end: date,
) -> tuple[pl.DataFrame, dict[str, Path]]:
    parts: dict[str, pl.DataFrame] = {}
    paths: dict[str, Path] = {}
    expected_inputs = {
        "predictions_sha256": common_manifest["sources"]["boosting_predictions"]["sha256"],
        "terminal_registry_sha256": common_manifest["terminal_entry_gate"]["registry_sha256"],
        "sector_history_sha256": sector_digest,
    }
    for source, digest in backtest["inputs"].items():
        source_path = Path(source)
        if source_path.name != "sector_champion_holdings.parquet":
            continue
        path = (repository_root / source_path).resolve()
        if not path.is_relative_to(repository_root.resolve()):
            raise ValueError("Sector allocation manifest path is outside its repository root.")
        manifest_path = path.with_name("manifest.json")
        manifest_key = str(source_path.with_name("manifest.json"))
        _verify_sha(manifest_path, backtest["input_manifests"][manifest_key])
        allocation = _read_manifest(manifest_path)
        strategy = allocation["strategy"]
        if allocation["policy_id"] != SECTOR_CHAMPION_POLICY_ID or strategy in parts:
            raise ValueError(f"Unexpected or duplicate sector allocation: {strategy}")
        if any(allocation["inputs"].get(key) != value for key, value in expected_inputs.items()):
            raise ValueError(f"Sector allocation and report lineage differ: {path}")
        _verify_sha(path, digest)
        parts[strategy] = _read_pending_basket(path, strategy=strategy, realized_end=realized_end)
        paths[f"allocation_manifest:{strategy}"] = manifest_path
        paths[f"allocation_holdings:{strategy}"] = path
    if set(parts) != {SECTOR_NATIVE, SECTOR_TREND}:
        raise ValueError("Sector report requires the two sealed allocations.")
    return pl.concat([parts[SECTOR_NATIVE], parts[SECTOR_TREND]]), paths


def _read_pending_basket(path: Path, *, strategy: str, realized_end: date) -> pl.DataFrame:
    frame = pl.read_parquet(path).filter(pl.col("holding_month") > realized_end)
    valid = frame.filter(
        (pl.col("strategy") == strategy)
        & (pl.col("decision_month") == realized_end)
        & (pl.col("holding_month") == pl.col("decision_month").dt.offset_by("1mo"))
        & (pl.col("target_status_1m") == "horizon_pending")
        & pl.col("realized_return").is_null()
        & ((pl.col("target_weight") - 1.0 / SECTOR_CHAMPION_TOP_N).abs() < 1e-12)
        & pl.col("Sector").is_not_null()
    )
    if (
        frame.height != SECTOR_CHAMPION_TOP_N
        or valid.height != SECTOR_CHAMPION_TOP_N
        or valid["Sector"].n_unique() != SECTOR_CHAMPION_TOP_N
        or valid["ticker"].n_unique() != SECTOR_CHAMPION_TOP_N
    ):
        raise ValueError(f"Expected five different sectors in the score-only basket: {path}")
    return valid.rename({"Sector": "sector"})


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected one manifest object in {path}")
    return manifest


def _verify_sha(path: Path, expected: str) -> None:
    if sha256_path(path) != expected:
        raise ValueError(f"Sector report input SHA-256 mismatch: {path}")
