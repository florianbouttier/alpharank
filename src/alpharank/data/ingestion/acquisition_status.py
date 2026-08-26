"""Run-scoped evidence that acquisition completed before publication gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from alpharank.data.ingestion.storage import write_json

ACQUISITION_STATUS_CONTRACT = "alpharank_source_acquisition_status_v1"


def persist_open_source_acquisition_status(
    *,
    run_dir: Path,
    run_id: str,
    source_refresh_contract: dict[str, object],
    run_failures: dict[str, list[dict[str, str]]],
    yahoo_prices: pl.DataFrame,
    yahoo_metadata_rows: int,
    yfinance_earnings: pl.DataFrame,
    sec_submissions: pl.DataFrame,
    sec_companyfacts: Sequence[pl.DataFrame],
    sec_filing_documents: Sequence[pl.DataFrame],
    simfin_fundamentals: Sequence[pl.DataFrame],
    yfinance_fundamentals: Sequence[pl.DataFrame],
) -> dict[str, Any]:
    """Persist final acquisition statuses immediately before publication gates."""

    price_gate_report = source_refresh_contract.get("price_revision_guard")
    if not isinstance(price_gate_report, dict):
        raise RuntimeError("Price revision gate report must be a JSON object")
    status = build_acquisition_status(
        run_id=run_id,
        source_rows={
            "yahoo_prices": yahoo_prices.height,
            "yahoo_metadata": yahoo_metadata_rows,
            "yfinance_earnings": yfinance_earnings.height,
            "sec_submissions": sec_submissions.height,
            "sec_companyfacts": sum(frame.height for frame in sec_companyfacts),
            "sec_filing_documents": sum(frame.height for frame in sec_filing_documents),
            "simfin_fundamentals": sum(frame.height for frame in simfin_fundamentals),
            "yfinance_fundamentals": sum(frame.height for frame in yfinance_fundamentals),
        },
        source_failures={
            "yfinance_earnings": run_failures["yfinance_earnings"],
            "sec_submissions": [
                failure
                for failure in run_failures["sec_filing"]
                if failure.get("dataset") == "earnings_sec_calendar"
            ],
            "sec_companyfacts": run_failures["sec_companyfacts"],
            "sec_filing_documents": run_failures["sec_filing"],
            "simfin_fundamentals": run_failures["simfin"],
        },
        price_gate_report=price_gate_report,
    )
    write_json(run_dir / "acquisition_status.json", status)
    write_json(run_dir / "source_refresh_contract.json", source_refresh_contract)
    write_json(run_dir / "run_failures.json", run_failures)
    return status


def build_acquisition_status(
    *,
    run_id: str,
    source_rows: Mapping[str, int],
    source_failures: Mapping[str, Sequence[Mapping[str, str]]],
    price_gate_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe fetched sources independently from the publication decision."""

    sources: list[dict[str, Any]] = []
    for source, row_count in source_rows.items():
        failures = list(source_failures.get(source, ()))
        status = _source_status(row_count=row_count, failure_count=len(failures))
        if source == "yahoo_prices" and not price_gate_report.get("passed", False):
            status = "downloaded_quarantined"
        elif source == "yahoo_prices" and price_gate_report.get(
            "resolved_provider_blocking_reasons"
        ):
            status = "downloaded_revisions_reconciled"
        sources.append(
            {
                "source": source,
                "status": status,
                "downloaded_rows": int(row_count),
                "failure_count": len(failures),
                "failure_examples": failures[:20],
            }
        )
    return {
        "contract": ACQUISITION_STATUS_CONTRACT,
        "run_id": run_id,
        "phase": "all_declared_sources_attempted_before_publication_decision",
        "price_publication_gate_passed": bool(price_gate_report.get("passed", False)),
        "price_publication_blocking_reasons": list(price_gate_report.get("blocking_reasons", ())),
        "sources": sources,
    }


def _source_status(*, row_count: int, failure_count: int) -> str:
    if failure_count and row_count:
        return "downloaded_with_failures"
    if failure_count:
        return "attempted_failed"
    if row_count:
        return "downloaded"
    return "completed_no_rows"
