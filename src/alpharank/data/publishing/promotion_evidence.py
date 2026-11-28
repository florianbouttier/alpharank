"""Validate the sealed replay evidence required to change a published composition."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from alpharank.governance_contracts.common import sha256_path

_ARTIFACTS = {
    "snapshot": ("lineage/manifest.json",),
    "legacy": (
        "data_input_manifest.json",
        "legacy_common_holdings.parquet",
        "legacy_common_monthly.parquet",
        "legacy_detailed_returns_polars.parquet",
    ),
    "boosting": ("manifest.json", "classification_h06/predictions.parquet"),
    "common": (
        "manifest.json",
        "comparison_common_holdings.parquet",
        "comparison_common_monthly.parquet",
        "boosting_live_score_holdings.parquet",
    ),
}

_MODEL_DATASETS = {
    "final_price": "US_Finalprice.parquet",
    "sp500_price": "SP500Price.parquet",
    "sp500_constituents": "SP500_Constituents.csv",
    "income_statement": "US_Income_statement.parquet",
    "balance_sheet": "US_Balance_sheet.parquet",
    "cash_flow": "US_Cash_flow.parquet",
    "earnings": "US_Earnings.parquet",
}


def seal_refresh_replay_inputs(inputs: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Fingerprint the exact artifacts used for outcomes and their run manifests."""
    result = {}
    for side in ("baseline", "candidate"):
        for role, filenames in _ARTIFACTS.items():
            key = f"{side}_{role}"
            directory = Path(inputs[key])
            result[key] = {name: sha256_path(directory / name) for name in filenames}
    return result


def validate_refresh_promotion_evidence(
    report_path: Path,
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, object]:
    """Bind a positive audit to both compositions and all replay evidence."""
    raw = report_path.read_bytes()
    report = json.loads(raw)
    stability = report.get("vintage_portfolio_stability") or {}
    if (
        report.get("contract_version") != 2
        or report.get("status") != "identical_historical_portfolios"
        or report.get("promotion_allowed_by_this_gate") is not True
        or report.get("common_replay_failure") is not None
        or stability.get("status") != "passed"
        or stability.get("exact_match") is not True
        or stability.get("date_source") != "baseline_live_portfolio"
    ):
        raise RuntimeError("Refresh replay evidence is not eligible for promotion")

    inputs = report["inputs"]
    if report.get("promotion_input_sha256") != seal_refresh_replay_inputs(inputs):
        raise RuntimeError("Refresh replay evidence has missing or changed artifact hashes")
    cutoff = date.fromisoformat(report["historical_cutoff"])
    baseline_end = _validate_side(inputs, "baseline", baseline)
    candidate_end = _validate_side(inputs, "candidate", candidate)
    if baseline_end > cutoff or cutoff > candidate_end:
        raise RuntimeError("Refresh replay evidence does not cover the published calendar")
    if date.fromisoformat(stability["decision_month"]) < cutoff:
        raise RuntimeError("Latest portfolio comparison predates the historical cutoff")
    return {
        "baseline_composition_id": baseline["composition_id"],
        "candidate_composition_id": candidate["composition_id"],
        "historical_cutoff": cutoff.isoformat(),
        "latest_decision_month": stability["decision_month"],
        "path": str(report_path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _validate_side(
    inputs: Mapping[str, str], side: str, expected: Mapping[str, Any]
) -> date:
    snapshot = _read_object(Path(inputs[f"{side}_snapshot"]) / "lineage/manifest.json")
    model_hashes = expected["output_sha256"]
    if (
        snapshot.get("composition_id") != expected["composition_id"]
        or snapshot.get("output_sha256") != model_hashes
    ):
        raise RuntimeError(f"Refresh replay evidence does not identify the {side} composition")

    legacy_path = Path(inputs[f"{side}_legacy"]) / "data_input_manifest.json"
    boosting_path = Path(inputs[f"{side}_boosting"]) / "manifest.json"
    for manifest_path, field in ((legacy_path, "datasets"), (boosting_path, "input_paths")):
        datasets = _read_object(manifest_path).get(field, {})
        for name, filename in _MODEL_DATASETS.items():
            if datasets.get(name, {}).get("sha256") != model_hashes[filename]:
                raise RuntimeError(f"Replay input hashes do not match the {side} composition")

    common = _read_object(Path(inputs[f"{side}_common"]) / "manifest.json")
    lineage = common.get("lineage_check", {})
    if (
        common.get("publication_eligible") is not True
        or lineage.get("passed") is not True
        or Path(lineage.get("left_manifest", "")).resolve() != legacy_path.resolve()
        or Path(lineage.get("right_manifest", "")).resolve() != boosting_path.resolve()
    ):
        raise RuntimeError(f"Common replay is not bound to publishable {side} model runs")
    return date.fromisoformat(common["calendar"]["end_holding_month"])


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload
