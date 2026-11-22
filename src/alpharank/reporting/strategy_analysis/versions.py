"""Describe economic backtest vintages independently of report rendering and price follow-up."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, TypedDict

import polars as pl

from alpharank.governance_contracts.common import sha256_path
from alpharank.reporting.strategy_analysis.dataset import AnalysisDataset, daily_valuation_policy
from alpharank.reporting.strategy_analysis.service import AnalysisQuery, query_analysis


class BacktestVersion(TypedDict):
    id: str
    economic_sha256: str
    replay_at_utc: str | None
    snapshot_at_utc: str | None
    price_as_of_date: str | None
    composition_id: str
    replay_git_commit: str | None
    start_date: str
    end_date: str
    last_close: str
    strategies: list[str]
    cost_policy: dict[str, Any]
    zero_cost_strategies: list[str]
    holding_month: str | None
    recent_last_close: str | None
    source_report_sha256: str | None
    publication_eligible: bool
    valuation_policy: str


def economic_identity(payload: dict[str, Any]) -> str:
    """Hash sealed methods/results, not UI timestamps, paths or a frozen-basket mark."""
    lineage = payload["lineage"]
    evidence = sorted(
        (row["role"], row["sha256"])
        for row in lineage["source_files"]
        if row["role"] != "portfolio_as_of_evidence"
    )
    contracts = {
        key: value for key, value in payload.get("contracts", {}).items() if key != "report_task"
    }
    basket = payload.get("current_portfolio", {})
    identity = {
        "composition_id": lineage["composition_id"],
        "replay_git_commit": lineage.get("replay_git_commit"),
        "comparison_profile": lineage.get("common_replay_profile"),
        "evidence": evidence,
        "contracts": contracts,
        "strategy_order": payload["strategy_order"],
        "monthly": payload["monthly"],
        "holdings": payload["holdings"],
        "basket": {key: basket.get(key) for key in ["decision_month", "holding_month", "holdings"]},
    }
    content = json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def _verified_run_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for reference in payload["lineage"]["source_files"]:
        role = reference["role"]
        if role in records:
            raise ValueError("Duplicate version evidence role: " + role)
        path = Path(reference["path"])
        if sha256_path(path) != reference["sha256"]:
            raise ValueError("Backtest version evidence SHA-256 differs: " + role)
        records[role] = (
            json.loads(path.read_text(encoding="utf-8"))
            if role in {"common_manifest", "snapshot_manifest"}
            else None
        )
    return records


def backtest_version(
    dataset: AnalysisDataset, *, source_report: Path | None = None
) -> BacktestVersion:
    """Describe the real replay cutoff without substituting the HTML generation date."""
    payload = dataset.payload
    records = _verified_run_metadata(payload)
    common, snapshot = records.get("common_manifest") or {}, records.get("snapshot_manifest") or {}
    identity = economic_identity(payload)
    costs = (
        pl.from_dicts(payload["monthly"], infer_schema_length=None)
        .group_by("strategy")
        .agg(pl.col("transaction_cost").abs().max().alias("maximum_cost"))
    )
    zero_cost = sorted(
        row["strategy"]
        for row in costs.iter_rows(named=True)
        if row["maximum_cost"] == 0 and row["strategy"] != dataset.benchmark_strategy
    )
    history = dataset.sources["history"]
    return {
        "id": "bt-" + identity[:16],
        "economic_sha256": identity,
        "replay_at_utc": common.get("runtime_provenance", {}).get("captured_at_utc"),
        "snapshot_at_utc": snapshot.get("generated_at"),
        "price_as_of_date": snapshot.get("data_freshness", {})
        .get("prices", {})
        .get("max_market_date"),
        "composition_id": payload["lineage"]["composition_id"],
        "replay_git_commit": payload["lineage"].get("replay_git_commit"),
        "start_date": history["min_date"],
        "end_date": history["max_date"],
        "last_close": history["last_close"],
        "strategies": list(payload["strategy_order"]),
        "cost_policy": payload.get("contracts", {}).get("transaction_cost", {}),
        "zero_cost_strategies": zero_cost,
        "holding_month": payload.get("current_portfolio", {}).get("holding_month"),
        "recent_last_close": dataset.sources.get("recent", {}).get("last_close"),
        "source_report_sha256": sha256_path(source_report) if source_report is not None else None,
        "publication_eligible": False,
        "valuation_policy": daily_valuation_policy(payload),
    }


def _holding_counts(
    left: AnalysisDataset, right: AnalysisDataset, start: date, end: date
) -> dict[str, Any]:
    keys = ["strategy", "holding_month", "ticker"]
    frames = []
    common = set(left.payload["strategy_order"]) & set(right.payload["strategy_order"])
    for dataset, side in [(left, "left"), (right, "right")]:
        frame = pl.from_dicts(dataset.payload["holdings"], infer_schema_length=None)
        frame = frame.filter(
            pl.col("strategy").is_in(common)
            & pl.col("holding_month").is_between(
                pl.lit(start.replace(day=1).isoformat()), pl.lit(end.replace(day=1).isoformat())
            )
        ).select(*keys, pl.col("target_weight").alias(side))
        if frame.select(keys).is_duplicated().any():
            raise ValueError("Version comparison requires unique strategy/month/ticker holdings.")
        frames.append(frame)
    joined = frames[0].join(frames[1], on=keys, how="full", coalesce=True, validate="1:1")
    changed = joined.filter(
        pl.col("left").is_null()
        | pl.col("right").is_null()
        | ((pl.col("left") - pl.col("right")).abs() > 1e-12)
    )
    return {
        "compared_positions": joined.height,
        "changed_positions": changed.height,
        "changed_baskets": changed.select("strategy", "holding_month").unique().height,
        "by_month": changed.group_by("strategy", "holding_month")
        .len(name="changed_positions")
        .sort("holding_month", "strategy")
        .to_dicts(),
        "scope": "Common strategies; selected months; titles and target weights only. Not a model replay or leakage test.",
    }


def compare_versions(
    left: AnalysisDataset, right: AnalysisDataset, query: AnalysisQuery
) -> dict[str, Any]:
    """Compare archived metrics on identical requested/effective bounds, never frozen follow-up."""
    if query.source not in {"history", "monthly_reference"} or query.sleeves or query.frontier:
        raise ValueError(
            "Version comparison only accepts the historical daily/monthly observations."
        )
    views = [query_analysis(dataset, query) for dataset in [left, right]]
    for key in ["anchor_date", "start_date", "end_date", "frequency"]:
        if views[0][key] != views[1][key]:
            raise ValueError("Versions do not share the same effective calendar: " + key)
    names = [name for name in left.payload["strategy_order"] if name in views[1]["series"]]
    left_contracts, right_contracts = (
        left.payload.get("contracts", {}),
        right.payload.get("contracts", {}),
    )
    contract_keys = (set(left_contracts) | set(right_contracts)) - {"report_task"}
    return {
        "requested_start": query.start.isoformat(),
        "requested_end": query.end.isoformat(),
        "anchor_date": views[0]["anchor_date"],
        "start_date": views[0]["start_date"],
        "end_date": views[0]["end_date"],
        "frequency": views[0]["frequency"],
        "metrics": [
            {
                "strategy": name,
                "left": views[0]["series"][name]["metrics"],
                "right": views[1]["series"][name]["metrics"],
            }
            for name in names
        ],
        "left_only_strategies": [
            name for name in left.payload["strategy_order"] if name not in names
        ],
        "right_only_strategies": [
            name for name in right.payload["strategy_order"] if name not in names
        ],
        "changed_contract_fields": sorted(
            key for key in contract_keys if left_contracts.get(key) != right_contracts.get(key)
        ),
        "snapshot_changed": left.payload["lineage"]["composition_id"]
        != right.payload["lineage"]["composition_id"],
        "code_changed": left.payload["lineage"].get("replay_git_commit")
        != right.payload["lineage"].get("replay_git_commit"),
        "holdings": _holding_counts(left, right, query.start, query.end),
        "warning": "Comparaison descriptive de runs archivés, pas un nouveau replay ni une preuve "
        "d'absence de fuite. Les données, le code et les politiques peuvent différer ; aucun écart "
        "n'est attribué automatiquement aux seules données.",
    }
