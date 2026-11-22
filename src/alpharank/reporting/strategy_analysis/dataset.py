"""Resolve one report's exact frozen data sources, never mutable latest pointers."""

from __future__ import annotations

import base64
import gzip
import json
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import polars as pl

from alpharank.governance_contracts.common import sha256_path
from alpharank.portfolio.daily import (
    DailyLedgerResult,
    build_daily_ledger,
    build_monthly_reference_ledger,
)
from alpharank.portfolio.daily.contracts import DailyValuationPolicy


@dataclass(frozen=True, slots=True)
class AnalysisDataset:
    """Frozen source payload and prepared read-only daily cohorts."""

    payload: dict[str, Any]
    ledgers: dict[str, pl.DataFrame]
    sources: dict[str, dict[str, str]]
    benchmark_strategy: str
    history: DailyLedgerResult | None = None


def daily_valuation_policy(payload: dict[str, Any]) -> DailyValuationPolicy:
    """Resolve only a declared versioned reporting policy, never a silent fill."""
    contract = payload.get("contracts", {}).get("daily_valuation")
    if contract is None:
        return "observed_only"
    if (
        not isinstance(contract, dict)
        or set(contract) != {"schema_version", "policy"}
        or contract["schema_version"] != 1
        or contract["policy"] not in {"observed_only", "last_observed_mark_v1"}
    ):
        raise ValueError("Unsupported daily valuation contract.")
    return cast(DailyValuationPolicy, contract["policy"])


def apply_valuation_policy(
    payload: dict[str, Any], policy: DailyValuationPolicy | None
) -> dict[str, Any]:
    """Create a distinct economic policy without changing any archived source field."""
    if policy is None:
        daily_valuation_policy(payload)
        return payload
    if policy not in {"observed_only", "last_observed_mark_v1"}:
        raise ValueError("Unknown daily valuation policy.")
    contracts = {
        key: value
        for key, value in payload.get("contracts", {}).items()
        if key != "daily_valuation"
    }
    if policy != "observed_only":
        contracts["daily_valuation"] = {"schema_version": 1, "policy": policy}
    return payload | {"contracts": contracts}


def read_performance_payload(path: Path) -> dict[str, Any]:
    """Decode the embedded source payload, independent of its historical UI library."""
    match = re.search(r'const PAYLOAD_GZIP_BASE64="([^"]+)"', path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError("The selected report has no sealed AlphaRank performance payload.")
    payload = json.loads(gzip.decompress(base64.b64decode(match.group(1))))
    if not isinstance(payload, dict) or not {
        "monthly",
        "holdings",
        "lineage",
        "strategies",
    }.issubset(payload):
        raise ValueError("The selected report does not expose the complete performance contract.")
    return payload


def _snapshot_prices(payload: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, str]]:
    lineage = payload["lineage"]
    records = [item for item in lineage["source_files"] if item["role"] == "snapshot_manifest"]
    if len(records) != 1:
        raise ValueError("Daily analysis requires one explicit sealed snapshot manifest.")
    record = records[0]
    manifest_path = Path(record["path"])
    if sha256_path(manifest_path) != record["sha256"]:
        raise ValueError("The snapshot manifest differs from the source report's sealed hash.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["composition_id"] != lineage["composition_id"]:
        raise ValueError("Daily analysis and frozen holdings must share the same composition.")
    frames, hashes = (
        [],
        {
            "composition_id": manifest["composition_id"],
            "snapshot_manifest_sha256": record["sha256"],
        },
    )
    for filename in ["US_Finalprice.parquet", "SP500Price.parquet"]:
        path = manifest_path.parent / filename
        digest = sha256_path(path)
        if digest != manifest["output_sha256"][filename]:
            raise ValueError(
                "Daily source price SHA-256 differs from the frozen snapshot: " + filename
            )
        frame = pl.read_parquet(path, columns=["ticker", "date", "adjusted_close"])
        day = (
            pl.col("date").str.to_date()
            if frame.schema["date"] == pl.String
            else pl.col("date").cast(pl.Date)
        )
        frames.append(frame.with_columns(day))
        hashes[filename + "_sha256"] = digest
    if set(frames[1]["ticker"].unique().to_list()) != {"SPY.US"}:
        raise ValueError("The explicit benchmark price artifact must contain only SPY.US.")
    return pl.concat([frames[0].filter(pl.col("ticker") != "SPY.US"), frames[1]]), hashes


def _recent_ledger(payload: dict[str, Any]) -> pl.DataFrame | None:
    recent = payload.get("recent_valuation")
    if recent is None:
        return None
    window = max(
        recent["windows"].values(),
        key=lambda item: len(item["series"][payload["strategy_order"][0]]),
    )
    rows = []
    sizes: dict[str, int] = {}
    for holding in payload["current_portfolio"]["holdings"]:
        sizes[holding["strategy"]] = sizes.get(holding["strategy"], 0) + 1
    for name in payload["strategy_order"]:
        for point in window["series"][name]:
            rows.append(
                {
                    "strategy": name,
                    "valuation_date": point["date"],
                    "wealth": point["wealth"],
                    "transaction_cost": 0.0,
                    "turnover": 0.0,
                    "n_positions": sizes.get(name, 1),
                    "maximum_position_weight": None,
                    "maximum_sector_weight": None,
                }
            )
    return (
        pl.from_dicts(rows, infer_schema_length=None)
        .with_columns(pl.col("valuation_date").str.to_date())
        .with_columns(pl.col("valuation_date").dt.truncate("1mo").alias("holding_month"))
        .sort("strategy", "valuation_date")
    )


def _history_sources(
    payload: dict[str, Any],
    history: DailyLedgerResult,
    source_hashes: dict[str, str],
) -> dict[str, dict[str, str]]:
    days = sorted(history.ledger["valuation_date"].unique().to_list())
    last_month = date.fromisoformat(payload["calendar"]["end"])
    history_end = last_month.replace(day=monthrange(last_month.year, last_month.month)[1])
    policy = daily_valuation_policy(payload)
    sources = {
        "history": {
            "label": "Backtest historique",
            "min_date": payload["calendar"]["start"],
            "max_date": history_end.isoformat(),
            "last_close": days[-1].isoformat(),
            **source_hashes,
            "valuation_policy": policy,
            "warning": (
                "Scénario estimé : seul le titre sans cours conserve son dernier cours connu "
                "jusqu'à la fin du mois. Ce n'est ni un prix observé ni une vente garantie. "
                "Positions et frais archivés inchangés."
                if policy == "last_observed_mark_v1"
                else "Reconstitution des paniers archivés, nette de leurs frais. "
                "Les clôtures manquantes restent des trous."
            ),
        }
    }
    sources["monthly_reference"] = {
        "label": "Référence mensuelle · repli",
        "min_date": payload["calendar"]["start"],
        "max_date": history_end.isoformat(),
        "last_close": days[-1].isoformat(),
        "composition_id": source_hashes["composition_id"],
        "warning": "Repli explicite sur les rendements mensuels scellés, risques à 12 observations/an. "
        "Aucun prix quotidien imputé ; ce mode conserve les hypothèses terminales de l'archive, "
        "sans les corriger ni mesurer les creux intra-mois.",
    }
    return sources


def prepare_analysis_dataset(payload: dict[str, Any]) -> AnalysisDataset:
    """Prepare a source-backed daily cohort without mutating the source report."""
    benchmarks = [item["label"] for item in payload["strategies"] if item["family"] == "Benchmark"]
    if len(benchmarks) != 1:
        raise ValueError("The source report must identify exactly one benchmark strategy.")
    prices, source_hashes = _snapshot_prices(payload)
    holdings = pl.from_dicts(payload["holdings"], infer_schema_length=None).with_columns(
        pl.col("holding_month").str.to_date(), pl.col("decision_month").str.to_date()
    )
    monthly = pl.from_dicts(payload["monthly"], infer_schema_length=None).with_columns(
        pl.col("holding_month").str.to_date(), pl.col("decision_month").str.to_date()
    )
    history = build_daily_ledger(
        holdings,
        monthly,
        prices,
        benchmark_strategy=benchmarks[0],
        valuation_policy=daily_valuation_policy(payload),
    )
    sources = _history_sources(payload, history, source_hashes)
    ledgers = {
        "history": history.ledger,
        "monthly_reference": build_monthly_reference_ledger(
            monthly, history.ledger, benchmark_strategy=benchmarks[0]
        ),
    }
    recent = _recent_ledger(payload)
    if recent is not None:
        ledgers["recent"] = recent
        sessions = sorted(recent["valuation_date"].unique().to_list())
        retrieved = (
            payload["recent_valuation"]
            .get("provenance", {})
            .get("retrieved_at", sessions[-1].isoformat())[:10]
        )
        sources["recent"] = {
            "label": "Suivi du panier figé",
            "min_date": sessions[1].isoformat(),
            "max_date": max(sessions[-1].isoformat(), retrieved),
            "last_close": sessions[-1].isoformat(),
            "warning": "Maintien hypothétique du panier prévu pour "
            + payload["current_portfolio"]["holding_month"][:7]
            + ", brut de frais, sans nouveau signal ni rééquilibrage.",
        }
    return AnalysisDataset(payload, ledgers, sources, benchmarks[0], history)
