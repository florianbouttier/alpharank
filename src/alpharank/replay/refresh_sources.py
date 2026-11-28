"""Source-level statuses for refreshes stopped before a candidate snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def failed_refresh_gate(run_dir: Path) -> dict[str, Any]:
    """Describe a rejected data gate using the inputs saved by acquisition."""
    for name in ("price_publication_guard", "price_revision_guard"):
        path = run_dir / f"{name}.json"
        if not path.is_file():
            continue
        price = _read_json(path)
        reasons = price.get("blocking_reasons", price.get("failure_reasons", []))
        if price.get("passed") is False or ("passed" not in price and reasons):
            return _failed_price_gate(name, price, reasons)
    path = run_dir / "historical_revision_guard.json"
    if path.is_file():
        fundamentals = _read_json(path)
        if fundamentals.get("historical_revisions_detected") is True and not (
            fundamentals.get("override_enabled") is True
            and fundamentals.get("approval_recorded") is True
        ):
            return {
                "name": path.stem,
                "passed": False,
                "reasons": ["unreviewed_historical_fundamental_revisions"],
                "cutoff_date": fundamentals["cutoff_date"],
                "guard_days": fundamentals["guard_days"],
                "blocked_datasets": fundamentals["blocked_datasets"],
                "datasets": fundamentals["datasets"],
            }
    raise ValueError(f"No failed refresh guard in {run_dir}")


def _failed_price_gate(name: str, guard: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    fields = (
        "candidate_rows",
        "candidate_tickers",
        "historical_daily_return_revisions_over_threshold",
        "historical_return_revision_tickers",
        "historical_return_availability_changes",
        "transition_factor_findings",
    )
    return {
        "name": name,
        "passed": False,
        "reasons": reasons,
        "policy": guard.get("policy", {}),
        "historical_return_revision_examples": guard.get("historical_return_revision_examples", []),
        **{field: guard.get(field) for field in fields},
    }


def blocked_refresh_source_statuses(failed_refresh_run: Path) -> list[dict[str, Any]]:
    """Describe every declared refresh source after a data gate fails."""

    acquisition_path = failed_refresh_run / "acquisition_status.json"
    if acquisition_path.is_file():
        acquisition = _read_json(acquisition_path)
        acquired = acquisition.get("sources")
        if not isinstance(acquired, list):
            raise ValueError(f"Invalid acquisition source statuses: {acquisition_path}")
        return [*acquired, *_retained_source_statuses(failed_refresh_run)]

    coverage = _read_json(failed_refresh_run / "price_validated_key_coverage.json")
    composition = _read_json(failed_refresh_run / "price_composition.json")
    raw_archive = coverage.get("raw_archive", {})
    statuses = [
        {
            "source": "yahoo_prices",
            "status": "downloaded_quarantined",
            "evidence_manifest": raw_archive.get("manifest_path"),
            "requested_active_tickers": composition.get("refreshable_active_ticker_count"),
            "resolved_active_rows": composition.get("active_yahoo_rows"),
            "provider_complete": coverage.get("provider_complete"),
            "definitive_resolution_passed": coverage.get("definitive_resolution", {}).get("passed"),
        },
        {
            "source": "eodhd_price_seed",
            "status": "retained_not_redownloadable",
            "preserved_rows": composition.get("preserved_history_rows"),
            "preserved_tickers": composition.get("preserved_history_tickers"),
            "reason": "Frozen historical evidence for inactive or delisted instruments.",
        },
        {
            "source": "previous_validated_open_source_prices",
            "status": "retained_by_vintage",
            "preserved_rows": composition.get("preserved_open_source_only_rows"),
            "preserved_tickers": composition.get("preserved_open_source_only_tickers"),
            "audited_carried_active_rows": composition.get("audited_carried_active_rows"),
            "audited_carried_active_tickers": composition.get("audited_carried_active_tickers"),
        },
        {
            "source": "sp500_constituent_registry",
            "status": "retained_reference_input",
            "reason": "This ingestion entrypoint consumes the validated registry as reference data.",
        },
    ]
    statuses.extend(_upstream_blocked_status(source) for source in _BLOCKED_SOURCES)
    return statuses


def _retained_source_statuses(failed_refresh_run: Path) -> list[dict[str, Any]]:
    composition = _read_json(failed_refresh_run / "price_composition.json")
    return [
        {
            "source": "eodhd_price_seed",
            "status": "retained_not_redownloadable",
            "preserved_rows": composition.get("preserved_history_rows"),
            "preserved_tickers": composition.get("preserved_history_tickers"),
        },
        {
            "source": "previous_validated_open_source_prices",
            "status": "retained_by_vintage",
            "preserved_rows": composition.get("preserved_open_source_only_rows"),
            "preserved_tickers": composition.get("preserved_open_source_only_tickers"),
            "audited_carried_active_rows": composition.get("audited_carried_active_rows"),
            "audited_carried_active_tickers": composition.get("audited_carried_active_tickers"),
        },
        {
            "source": "sp500_constituent_registry",
            "status": "retained_reference_input",
        },
    ]


def _upstream_blocked_status(source: str) -> dict[str, str]:
    return {
        "source": source,
        "status": "not_started_blocked_upstream",
        "reason": "The price candidate failed before the financial acquisition stage.",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing blocked refresh evidence: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Blocked refresh evidence must contain an object: {path}")
    return payload


_BLOCKED_SOURCES = (
    "yahoo_metadata",
    "sec_companyfacts",
    "sec_submissions",
    "sec_filing_documents",
    "simfin_fundamentals",
    "yfinance_fundamentals",
)
